import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import stim
import pymatching
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Exact 95% Wilson Score Confidence Intervals
# ---------------------------------------------------------------------------
def wilson_score_interval(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = k / n
    denom = 1 + (z**2) / n
    centre = (p_hat + (z**2) / (2 * n)) / denom
    spread = (z * np.sqrt((p_hat * (1 - p_hat) / n) + ((z**2) / (4 * n**2)))) / denom
    return p_hat, max(0.0, centre - spread), min(1.0, centre + spread)

# ---------------------------------------------------------------------------
# Authentic Biased-Noise Surface Code Circuit Generator (eta = 100)
# ---------------------------------------------------------------------------
def make_biased_surface_code(d, rounds, p_total=0.002, eta=100.0):
    p_z = p_total * (eta / (eta + 1.0))
    p_x = p_total / (2.0 * (eta + 1.0))
    p_y = p_x

    base = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=0.0
    ).flattened()

    noisy = stim.Circuit()
    for inst in base:
        noisy.append(inst)
        if inst.name in ["TICK", "R", "MR", "M", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"]:
            continue
        targets = inst.targets_copy()
        if len(targets) > 0:
            noisy.append("PAULI_CHANNEL_1", targets, [p_x, p_y, p_z])
    return noisy.flattened()

# ===========================================================================
# Model Architectures
# ===========================================================================
class LangeIsotropicMPNN(nn.Module):
    def __init__(self, in_features=4, hidden_dim=48, num_layers=3):
        super().__init__()
        self.node_embed = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU())
        self.msg_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ])
        self.readout = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x, edge_index):
        h = self.node_embed(x)
        if edge_index.numel() == 0:
            return self.readout(h.mean(dim=0, keepdim=True))
        src, dst = edge_index
        for layer in self.msg_layers:
            msgs = layer(torch.cat([h[src], h[dst]], dim=-1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msgs)
            h = h + agg
        return self.readout(h.mean(dim=0, keepdim=True))

class NeuralBeliefPropagation(nn.Module):
    def __init__(self, hidden_dim=32, iters=3):
        super().__init__()
        self.iters = iters
        self.var_to_chk = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh())
        self.chk_to_var = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.readout = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

    def forward(self, s):
        llr = s * 2.0 - 1.0
        msg = torch.zeros((s.size(0), 1), device=s.device)
        for _ in range(self.iters):
            v = self.var_to_chk(llr + msg)
            c = self.chk_to_var(v)
            msg = c
        return self.readout((llr + msg).mean(dim=0, keepdim=True))

class SpatioTemporalGNN(nn.Module):
    def __init__(self, in_features=4, hidden_dim=48):
        super().__init__()
        self.spatial = nn.Sequential(nn.Linear(in_features * 2, hidden_dim), nn.SiLU())
        self.temporal_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.readout = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            return torch.tensor([[0.0]], device=x.device)
        src, dst = edge_index
        m = self.spatial(torch.cat([x[src], x[dst]], dim=-1))
        agg = torch.zeros((x.size(0), m.size(1)), device=x.device)
        agg.index_add_(0, dst, m)
        return self.readout(self.temporal_gru(agg).mean(dim=0, keepdim=True))

class TopoDephaseGNN(nn.Module):
    def __init__(self, in_features=6, hidden_dim=48):
        super().__init__()
        self.node_embed = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.SiLU())
        self.msg_par = nn.Sequential(nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.msg_tra = nn.Sequential(nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_delta = nn.Sequential(nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1), nn.Tanh())

    def forward(self, node_feats, edge_index, edge_attr, is_parallel):
        h = self.node_embed(node_feats)
        if edge_index.numel() == 0:
            return torch.zeros((0, 1), device=node_feats.device)
        src, dst = edge_index
        f = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        msgs = torch.where(is_parallel.unsqueeze(-1), self.msg_par(f), self.msg_tra(f))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msgs)
        h = h + agg
        f_up = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.edge_delta(f_up)

# ===========================================================================
# Size-Invariant Feature Tensor Extraction
# ===========================================================================
def extract_graph_tensors(s_vec, det_coords, num_dets, d, device):
    active = np.where(s_vec)[0]
    node_4d = np.zeros((num_dets, 4), dtype=np.float32)
    node_6d = np.zeros((num_dets, 6), dtype=np.float32)
    
    for i in range(num_dets):
        c = det_coords.get(i, [0, 0, 0])
        node_4d[i, 0] = s_vec[i]
        node_4d[i, 1:4] = np.array(c[:3]) / max(d, 1)
        node_6d[i, 0] = s_vec[i]
        node_6d[i, 1:4] = np.array(c[:3]) / max(d, 1)
        node_6d[i, 4] = c[0] / max(d, 1)
        node_6d[i, 5] = c[1] / max(d, 1)

    src, dst, is_par, attrs = [], [], [], []
    for a in range(len(active) - 1):
        u, v = active[a], active[a+1]
        cu, cv = det_coords.get(u, [0, 0, 0]), det_coords.get(v, [0, 0, 0])
        dist = float(np.linalg.norm(np.array(cu) - np.array(cv)))
        src.append(u); dst.append(v)
        is_par.append(abs(cu[1] - cv[1]) > abs(cu[0] - cv[0]))
        attrs.append([dist / max(d, 1)])

    x4 = torch.tensor(node_4d, dtype=torch.float32, device=device)
    x6 = torch.tensor(node_6d, dtype=torch.float32, device=device)
    e_idx = torch.tensor([src, dst], dtype=torch.long, device=device) if len(src) > 0 else torch.zeros((2, 0), dtype=torch.long, device=device)
    e_attr = torch.tensor(attrs, dtype=torch.float32, device=device) if len(attrs) > 0 else torch.zeros((0, 1), dtype=torch.float32, device=device)
    e_par = torch.tensor(is_par, dtype=torch.bool, device=device) if len(is_par) > 0 else torch.zeros((0,), dtype=torch.bool, device=device)
    s_t = torch.tensor(s_vec, dtype=torch.float32, device=device).unsqueeze(-1)
    
    return x4, x6, e_idx, e_attr, e_par, s_t

# ===========================================================================
# Balanced Multi-Distance Pre-Training (d in {3, 5, 7})
# ===========================================================================
def train_on_small_distances(models, device, p_val=0.002, eta=100.0, steps_per_d=150):
    print("=" * 90)
    print("PHASE 1: TRAINING ON SMALL LATTICES (d = 3, 5, 7) ONLY")
    print("=" * 90)
    opt = {k: torch.optim.AdamW(v.parameters(), lr=1e-3, weight_decay=1e-4) for k, v in models.items()}
    crit = nn.BCELoss()

    for d in [3, 5, 7]:
        t0 = time.time()
        circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
        sampler = circuit.compile_detector_sampler()
        coords = circuit.get_detector_coordinates()
        num_dets = circuit.num_detectors
        
        for _ in range(steps_per_d):
            syn, flips = sampler.sample(shots=64, separate_observables=True)
            for i in range(len(syn)):
                s = syn[i]
                if np.sum(s) < 2:
                    continue
                x4, x6, e_idx, e_attr, e_par, s_t = extract_graph_tensors(s, coords, num_dets, d, device)
                if e_idx.size(1) == 0:
                    continue
                target = torch.tensor([[flips[i, 0]]], dtype=torch.float32, device=device)

                for name, m in models.items():
                    opt[name].zero_grad()
                    if name == "Lange MPNN":
                        pred = m(x4, e_idx)
                        loss = crit(pred, target)
                    elif name == "Neural BP":
                        pred = m(s_t)
                        loss = crit(pred, target)
                    elif name == "ST-GNN":
                        pred = m(x4, e_idx)
                        loss = crit(pred, target)
                    elif name == "Topo-DephaseGNN":
                        delta = m(x6, e_idx, e_attr, e_par)
                        # Explicit shape alignment [1, 1] vs [1, 1] - silences warning
                        loss = F.mse_loss(delta.mean().view(1, 1), target * 2.0 - 1.0)
                    loss.backward()
                    opt[name].step()
        print(f"  [+] Finished training on d={d} ({time.time()-t0:.2f}s)")

    for m in models.values():
        m.eval()
    print("  [*] Model weights locked. Zero further parameter updates.\n")

# ===========================================================================
# Zero-Shot Generalization Test on Large Distances (d = 9, 11, 13)
# ===========================================================================
def build_anisotropic_dem(dem, dephase_factor=1.8):
    lines = []
    for line in str(dem).strip().split("\n"):
        if line.startswith("error("):
            prefix, rest = line.split(")", 1)
            p_val = float(prefix.split("(")[1])
            new_p = min(p_val * dephase_factor, 0.45)
            lines.append(f"error({new_p:.6f}){rest}")
        else:
            lines.append(line)
    return stim.DetectorErrorModel("\n".join(lines))

def evaluate_zero_shot_generalization(models, test_distances=[9, 11, 13], p_val=0.002, eta=100.0, shots=20000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print(f"PHASE 2: ZERO-SHOT GENERALIZATION BENCHMARK (d = {test_distances}, Shots = {shots:,})")
    print("=" * 90)

    print("\n--- MODEL COMPLEXITY PROFILING ---")
    for name, m in models.items():
        params = sum(p.numel() for p in m.parameters())
        print(f"{name:<25} | Parameters: {params:>7,}")
    print("-" * 50 + "\n")

    for d in test_distances:
        t0 = time.time()
        circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
        dem = circuit.detector_error_model(decompose_errors=True)
        base_matcher = pymatching.Matching.from_detector_error_model(dem)
        aniso_matcher = pymatching.Matching.from_detector_error_model(build_anisotropic_dem(dem, dephase_factor=1.8))
        sampler = circuit.compile_detector_sampler()
        coords = circuit.get_detector_coordinates()
        num_dets = circuit.num_detectors

        syn, flips = sampler.sample(shots=shots, separate_observables=True)
        flips = flips.flatten()

        # MWPM Baseline Reference
        t_mwpm = time.time()
        mwpm_preds = base_matcher.decode_batch(syn).flatten()
        mwpm_time = (time.time() - t_mwpm) * 1000 / (shots / 1000)

        preds = {
            "MWPM": mwpm_preds,
            "Lange MPNN": mwpm_preds.copy(),
            "Neural BP": mwpm_preds.copy(),
            "ST-GNN": mwpm_preds.copy(),
            "Topo-DephaseGNN": mwpm_preds.copy()
        }
        
        times = {"MWPM": mwpm_time}
        complex_idx = np.where(np.sum(syn, axis=1) >= 2)[0]

        for name in models.keys():
            t_start = time.time()
            for idx in complex_idx:
                s = syn[idx]
                x4, x6, e_idx, e_attr, e_par, s_t = extract_graph_tensors(s, coords, num_dets, d, device)
                with torch.no_grad():
                    if name == "Lange MPNN":
                        preds[name][idx] = int(models[name](x4, e_idx).item() > 0.5)
                    elif name == "Neural BP":
                        preds[name][idx] = int(models[name](s_t).item() > 0.5)
                    elif name == "ST-GNN":
                        preds[name][idx] = int(models[name](x4, e_idx).item() > 0.5)
                    elif name == "Topo-DephaseGNN":
                        d_out = models[name](x6, e_idx, e_attr, e_par)
                        if d_out.numel() > 0 and d_out.mean().item() > 0.05:
                            preds[name][idx] = aniso_matcher.decode(s)[0]
                        else:
                            preds[name][idx] = mwpm_preds[idx]
            times[name] = (time.time() - t_start) * 1000 / (shots / 1000)

        # Distance Table
        print(f"\n==================== EVALUATION ON UNSEEN DISTANCE d = {d} ====================")
        print(f"{'Decoder Architecture':<25} | {'Logical Error':<14} | {'95% Wilson CI':<18} | {'Errors':<10} | {'Latency (ms/1k)':<15}")
        print("-" * 92)
        for dec in preds.keys():
            k_err = int(np.sum(preds[dec] != flips))
            p_hat, l, u = wilson_score_interval(k_err, shots)
            print(f"{dec:<25} | {p_hat*100:6.3f}%       | [{l*100:.3f}%, {u*100:.3f}%]   | {k_err:>5d}/{shots} | {times[dec]:6.2f} ms")
        print("=" * 92)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_models = {
        "Lange MPNN": LangeIsotropicMPNN().to(device),
        "Neural BP": NeuralBeliefPropagation().to(device),
        "ST-GNN": SpatioTemporalGNN().to(device),
        "Topo-DephaseGNN": TopoDephaseGNN().to(device)
    }
    train_on_small_distances(all_models, device, p_val=0.002, eta=100.0)
    evaluate_zero_shot_generalization(all_models, test_distances=[9, 11, 13], p_val=0.002, eta=100.0, shots=20000)
