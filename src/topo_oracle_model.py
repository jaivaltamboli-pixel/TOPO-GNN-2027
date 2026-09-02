import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import heapq
import time
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph

VIRTUAL_BOUNDARY = -1

def standardize_edge(u, v, bnd_z_idx, bnd_x_idx):
    """Maps internal graph boundary indices to standardized virtual boundary -1."""
    u_clean = VIRTUAL_BOUNDARY if (u == bnd_z_idx or u == bnd_x_idx or u == -1) else int(u)
    v_clean = VIRTUAL_BOUNDARY if (v == bnd_z_idx or v == bnd_x_idx or v == -1) else int(v)
    if u_clean == VIRTUAL_BOUNDARY and v_clean != VIRTUAL_BOUNDARY:
        return (v_clean, VIRTUAL_BOUNDARY)
    elif v_clean == VIRTUAL_BOUNDARY and u_clean != VIRTUAL_BOUNDARY:
        return (u_clean, VIRTUAL_BOUNDARY)
    else:
        return tuple(sorted((u_clean, v_clean)))

def build_parity_expanded_graph(dem, num_dets, coords, d):
    """
    Extracts complete DEM graph and builds state graph (v, q) with q in {0, 1}.
    """
    edge_dict_raw, bnd_z_idx, bnd_x_idx, _ = extract_complete_dem_graph(dem, num_dets, coords, d)
    
    # Standardized edge dictionary keyed by canonical (u, v)
    edge_dict = {}
    adj = {}  # node -> list of (neighbor, weight, obs_parity, edge_tuple)

    def add_adj(u, v, w, obs_bit, canon_e):
        if u not in adj:
            adj[u] = []
        adj[u].append((v, w, obs_bit, canon_e))

    for (u, v), props in edge_dict_raw.items():
        canon = standardize_edge(u, v, bnd_z_idx, bnd_x_idx)
        w = float(props["weight"])
        obs_bit = 1 if props.get("has_obs", False) else 0
        
        # Keep lowest weight if duplicate DEM edge mechanisms map to same pair
        if canon not in edge_dict or w < edge_dict[canon]["weight"]:
            edge_dict[canon] = {"weight": w, "has_obs": bool(obs_bit)}

    # Build adjacency with resolved properties
    for canon, props in edge_dict.items():
        u, v = canon
        w = props["weight"]
        obs_bit = 1 if props["has_obs"] else 0
        add_adj(u, v, w, obs_bit, canon)
        if v != u:
            add_adj(v, u, w, obs_bit, canon)

    return edge_dict, adj, bnd_z_idx, bnd_x_idx

def find_exact_logical_reference_chain(adj, num_dets):
    """
    Phase A: Computes Dijkstra on (v, q) state graph from (BOUNDARY, 0) to (BOUNDARY, 1).
    Guarantees:
      1. partial(R_L) == 0 in detector space (endpoints only at virtual boundary -1).
      2. obs(R_L) == 1 strictly.
    """
    start_node = VIRTUAL_BOUNDARY
    # Priority queue: (dist, current_node, current_parity, path_edges)
    pq = [(0.0, start_node, 0, [])]
    visited = {}

    while pq:
        dist, u, q, path = heapq.heappop(pq)

        state = (u, q)
        if state in visited and visited[state] <= dist:
            continue
        visited[state] = dist

        # Target condition: reached boundary with odd observable parity (path length > 0)
        if u == VIRTUAL_BOUNDARY and q == 1 and len(path) > 0:
            return set(path), dist

        for v, w, obs_bit, canon_e in adj.get(u, []):
            next_q = q ^ obs_bit
            next_state = (v, next_q)
            next_dist = dist + w
            if next_state not in visited or next_dist < visited[next_state]:
                heapq.heappush(pq, (next_dist, v, next_q, path + [canon_e]))

    raise RuntimeError("Failed to find valid logical reference chain R_L in DEM!")

def compute_chain_boundary(chain_edges, num_dets):
    """Computes partial(C) mod 2 in detector space (excluding virtual boundary)."""
    bnd = np.zeros(num_dets, dtype=np.uint8)
    for u, v in chain_edges:
        if u != VIRTUAL_BOUNDARY and u < num_dets:
            bnd[u] ^= 1
        if v != VIRTUAL_BOUNDARY and v < num_dets:
            bnd[v] ^= 1
    return bnd

def compute_chain_observable(chain_edges, edge_dict):
    """Computes obs(C) mod 2."""
    obs = 0
    for e in chain_edges:
        if edge_dict.get(e, {}).get("has_obs", False):
            obs ^= 1
    return obs

def compute_chain_weight(chain_edges, edge_dict):
    """Computes classical DEM weight W(C)."""
    return sum(edge_dict.get(e, {}).get("weight", 4.5) for e in chain_edges)

def run_phase_a_d_audit(distances=[3, 5, 7, 9], p_val=0.002, eta=100.0, audit_shots=100000):
    print("=" * 115)
    print(f"PHASE A–D EXACT DUAL-COSET OPPORTUNITY AUDIT ({audit_shots:,} shots/distance, p={p_val}, Bias eta={eta})")
    print("=" * 115 + "\n")

    for d in distances:
        t0 = time.time()
        circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
        dem = circuit.detector_error_model(decompose_errors=True)
        coords = circuit.get_detector_coordinates()
        num_dets = circuit.num_detectors

        # 1. Build parity state graph and exact R_L
        edge_dict, adj, bnd_z_idx, bnd_x_idx = build_parity_expanded_graph(dem, num_dets, coords, d)
        R_L, w_ref = find_exact_logical_reference_chain(adj, num_dets)

        # 2. Verify R_L Invariants
        bnd_R_L = compute_chain_boundary(R_L, num_dets)
        obs_R_L = compute_chain_observable(R_L, edge_dict)
        assert np.all(bnd_R_L == 0), f"d={d}: Reference chain R_L has non-zero detector boundary: {np.where(bnd_R_L > 0)[0]}"
        assert obs_R_L == 1, f"d={d}: Reference chain R_L does not have odd observable parity!"

        matcher = pymatching.Matching.from_detector_error_model(dem)
        sampler = circuit.compile_detector_sampler()
        syn, flips = sampler.sample(shots=audit_shots, separate_observables=True)
        flips = flips.flatten().astype(np.int64)

        # Trackers
        passed_invariants = 0
        mwpm_errors = 0
        recoverable_mwpm_errors = 0
        w_gap_list = []

        for idx in range(audit_shots):
            s = syn[idx].astype(np.uint8)
            y_true = flips[idx]

            # Blossom global match C_A
            edges_a_raw = matcher.decode_to_edges_array(s)
            C_A = set(standardize_edge(int(e[0]), int(e[1]), bnd_z_idx, bnd_x_idx) for e in edges_a_raw)

            # Exact candidate C_B = C_A XOR R_L
            C_B = C_A.symmetric_difference(R_L)

            # Phase C Invariant Verification
            bnd_A = compute_chain_boundary(C_A, num_dets)
            bnd_B = compute_chain_boundary(C_B, num_dets)
            obs_A = compute_chain_observable(C_A, edge_dict)
            obs_B = compute_chain_observable(C_B, edge_dict)

            assert np.array_equal(bnd_A, s), f"Shot {idx}: Chain A boundary != syndrome!"
            assert np.array_equal(bnd_B, s), f"Shot {idx}: Chain B boundary != syndrome!"
            assert obs_A != obs_B, f"Shot {idx}: Homological degeneracy broken (obs_A == obs_B == {obs_A})!"
            passed_invariants += 1

            # Candidate weights
            w_A = compute_chain_weight(C_A, edge_dict)
            w_B = compute_chain_weight(C_B, edge_dict)
            w_gap_list.append(w_B - w_A)

            if obs_A != y_true:
                mwpm_errors += 1
                # Check if alternative candidate C_B contains true logical class
                if obs_B == y_true:
                    recoverable_mwpm_errors += 1

        mwpm_rate = (mwpm_errors / audit_shots) * 100.0
        rec_rate = (recoverable_mwpm_errors / mwpm_errors * 100.0) if mwpm_errors > 0 else 0.0
        avg_gap = np.mean(w_gap_list)
        min_gap = np.min(w_gap_list)

        print(f"============================== DISTANCE d = {d:2d} ({time.time()-t0:.2f}s) ==============================")
        print(f"  Invariant Checks Passed:             {passed_invariants:,}/{audit_shots:,} (100.00%)")
        print(f"  Exact Reference Weight W(R_L):       {w_ref:.3f} ({len(R_L)} edges)")
        print(f"  Classical MWPM Logical Errors:       {mwpm_errors:>5d}/{audit_shots:,} ({mwpm_rate:6.3f}%)")
        print(f"  Recoverable Failures (C_B is True):  {recoverable_mwpm_errors:>5d}/{mwpm_errors:>5d} ({rec_rate:6.2f}%)")
        print(f"  Mean Classical Weight Gap (W_B-W_A): {avg_gap:.3f} (min: {min_gap:.3f})")
        print("-" * 115 + "\n")

    print("=" * 115)
    print("PHASE A–D COMPLETE: Opportunity ceiling confirmed. Ready for neural candidate ranker.")
    print("=" * 115 + "\n")

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationalMessageLayer(nn.Module):
    def __init__(self, hidden_dim=64, in_edge_dim=4):
        super().__init__()
        msg_dim = hidden_dim * 2 + in_edge_dim + 1
        self.msg_mlp = nn.Sequential(nn.Linear(msg_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.node_update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index, edge_attr, is_par):
        if edge_index.shape[1] == 0:
            return h
        src, dst = edge_index[0].long(), edge_index[1].long()
        msg_input = torch.cat([h[src], h[dst], edge_attr, is_par], dim=-1)
        messages = self.msg_mlp(msg_input)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)
        return self.norm(h + self.node_update(torch.cat([h, agg], dim=-1)))


class MultiscaleTopoOracle(nn.Module):
    def __init__(self, in_node_dim=6, in_edge_dim=4, hidden_dim=64, num_layers=6, bins=3):
        super().__init__()
        self.bins = bins
        local_layers = num_layers // 2
        coarse_layers = num_layers - local_layers
        
        self.node_embed = nn.Sequential(nn.Linear(in_node_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.local_layers = nn.ModuleList([RelationalMessageLayer(hidden_dim, in_edge_dim) for _ in range(local_layers)])
        
        self.pool_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.coarse_layers = nn.ModuleList([RelationalMessageLayer(hidden_dim, in_edge_dim) for _ in range(coarse_layers)])
        
        self.global_attn = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + in_edge_dim + 1, hidden_dim), 
            nn.GELU(), 
            nn.Linear(hidden_dim, 1)
        )
        
        self.chain_energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.logical_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x6, edge_index, edge_attr, is_par, mask_diff, batch_map=None, num_graphs=None):
        h = self.node_embed(x6)
        src, dst = edge_index[0].long(), edge_index[1].long()
        for layer in self.local_layers:
            h = layer(h, edge_index, edge_attr, is_par)
            
        quant = (x6[:, 0:3] * self.bins).long().clamp(0, self.bins - 1)
        cell_id = quant[:, 0] * (self.bins**2) + quant[:, 1] * self.bins + quant[:, 2]
        cluster_id = batch_map * (self.bins**3) + cell_id
        _, cluster_idx = torch.unique(cluster_id, return_inverse=True)
        num_clusters = cluster_idx.max().item() + 1
        
        h_pool_in = self.pool_mlp(h)
        h_coarse = torch.zeros((num_clusters, h.shape[1]), device=h.device)
        h_coarse.index_add_(0, cluster_idx, h_pool_in)
        
        c_src, c_dst = cluster_idx[src], cluster_idx[dst]
        mask = c_src != c_dst
        super_edge_index = torch.stack([c_src[mask], c_dst[mask]], dim=0) if mask.any() else torch.empty((2, 0), dtype=torch.long, device=h.device)
        super_edge_attr = edge_attr[mask]
        super_is_par = is_par[mask]
        
        for layer in self.coarse_layers:
            h_coarse = layer(h_coarse, super_edge_index, super_edge_attr, super_is_par)
            
        attn_weights = self.global_attn(h_coarse)
        h_coarse_attn = h_coarse * attn_weights
        
        coarse_batch_map = torch.zeros(num_clusters, dtype=torch.long, device=h.device)
        coarse_batch_map[cluster_idx] = batch_map
        
        global_h = torch.zeros((num_graphs, h.shape[1]), device=h.device)
        global_h.index_add_(0, coarse_batch_map, h_coarse_attn)
        
        h_unpooled = h_coarse[cluster_idx]
        h_global_bcast = global_h[batch_map]
        
        h_combined = h + h_unpooled + h_global_bcast
        
        edge_feat = torch.cat([h_combined[src], h_combined[dst], edge_attr, is_par], dim=-1)
        edge_logits = self.edge_scorer(edge_feat)
        
        pred_delta_w = self.chain_energy_head(global_h)
        logical_logits = self.logical_head(global_h)
        
        return edge_logits, pred_delta_w, logical_logits

def physics_informed_loss(edge_logits, pred_delta_w, logical_logits, target_edges, target_delta_w, target_logical, mask_diff):
    L_logical = F.binary_cross_entropy_with_logits(logical_logits, target_logical)
    L_chain = F.mse_loss(pred_delta_w, target_delta_w)
    
    cycle_mask = (mask_diff.abs() > 0).float()
    L_topology = F.binary_cross_entropy_with_logits(edge_logits, target_edges, weight=cycle_mask) if target_edges is not None else torch.tensor(0.0, device=edge_logits.device)
    
    L_safety = torch.mean( F.relu( 0.75 - torch.abs(logical_logits) ) * torch.exp(-torch.abs(target_delta_w)) )
    
    return 1.0 * L_logical + 0.1 * L_chain + 1.0 * L_topology + 0.5 * L_safety

if __name__ == "__main__":
    run_phase_a_d_audit()
