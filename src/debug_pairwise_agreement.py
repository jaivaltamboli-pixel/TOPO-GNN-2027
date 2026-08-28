import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import stim
import pymatching
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
import itertools

def wilson_score_interval(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0, 0.0
    z = norm.ppf(1 - (1 - confidence) / 2)
    p_hat = k / n
    denom = 1 + (z**2) / n
    centre = (p_hat + (z**2) / (2 * n)) / denom
    spread = (z * np.sqrt((p_hat * (1 - p_hat) / n) + ((z**2) / (4 * n**2)))) / denom
    return p_hat, max(0.0, centre - spread), min(1.0, centre + spread)

def make_biased_surface_code(d=5, rounds=5, p_total=0.002, eta=100.0):
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

# --- Model Definitions with Architecture-Specific Inductive Biases ---
class LangeIsotropicMPNN(nn.Module):
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.node_embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.msg = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x, edge_index):
        h = self.node_embed(x)
        if edge_index.numel() == 0:
            return torch.tensor([[0.0]], device=x.device)
        src, dst = edge_index
        m = self.msg(torch.cat([h[src], h[dst]], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        return self.head((h + agg).mean(dim=0, keepdim=True))

class NeuralBP(nn.Module):
    def __init__(self, hidden=32, iters=3):
        super().__init__()
        self.iters = iters
        self.f = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

    def forward(self, s):
        x = s * 2.0 - 1.0
        msg = torch.zeros_like(x)
        for _ in range(self.iters):
            msg = self.f(x + msg)
        return self.head((x + msg).mean(dim=0, keepdim=True))

class STGNN(nn.Module):
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Linear(in_dim * 2, hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            return torch.tensor([[0.0]], device=x.device)
        src, dst = edge_index
        m = self.conv(torch.cat([x[src], x[dst]], dim=-1))
        agg = torch.zeros((x.size(0), m.size(1)), device=x.device)
        agg.index_add_(0, dst, m)
        return self.head(self.gru(agg).mean(dim=0, keepdim=True))

class TopoDephaseGNN(nn.Module):
    def __init__(self, in_dim=6, hidden=64):
        super().__init__()
        self.node_embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU())
        self.msg_par = nn.Sequential(nn.Linear(hidden * 2 + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.msg_tra = nn.Sequential(nn.Linear(hidden * 2 + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.head = nn.Sequential(nn.Linear(hidden * 2, 1), nn.Sigmoid())

    def forward(self, x, edge_index, edge_attr, is_par):
        h = self.node_embed(x)
        if edge_index.numel() == 0:
            return torch.tensor([[0.0]], device=x.device)
        src, dst = edge_index
        f = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        msgs = torch.where(is_par.unsqueeze(-1), self.msg_par(f), self.msg_tra(f))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msgs)
        pool = torch.cat([h[src].mean(dim=0, keepdim=True), h[dst].mean(dim=0, keepdim=True)], dim=-1)
        return self.head(pool)

# --- Balanced Batch Training ---
def train_balanced(models, circuit, device, steps=350):
    print("[*] Training models with balanced active error chain sampling...")
    opt = {k: torch.optim.AdamW(v.parameters(), lr=1.5e-3) for k, v in models.items()}
    crit = nn.BCELoss()
    sampler = circuit.compile_detector_sampler()
    det_coords = circuit.get_detector_coordinates()
    num_dets = circuit.num_detectors

    for _ in range(steps):
        syn, flips = sampler.sample(shots=64, separate_observables=True)
        for i in range(len(syn)):
            s_vec = syn[i]
            active = np.where(s_vec)[0]
            if len(active) < 2:
                continue

            node_4d = np.zeros((num_dets, 4), dtype=np.float32)
            node_6d = np.zeros((num_dets, 6), dtype=np.float32)
            for d_idx in range(num_dets):
                c = det_coords.get(d_idx, [0, 0, 0])
                node_4d[d_idx, 0] = s_vec[d_idx]
                node_4d[d_idx, 1:4] = c[:3]
                node_6d[d_idx, 0] = s_vec[d_idx]
                node_6d[d_idx, 1:4] = c[:3]
                node_6d[d_idx, 4] = c[0]
                node_6d[d_idx, 5] = c[1]

            src, dst, is_par, attrs = [], [], [], []
            for a in range(len(active) - 1):
                u, v = active[a], active[a+1]
                cu, cv = det_coords.get(u, [0,0,0]), det_coords.get(v, [0,0,0])
                src.append(u); dst.append(v)
                is_par.append(abs(cu[1] - cv[1]) > abs(cu[0] - cv[0]))
                attrs.append([float(np.linalg.norm(np.array(cu) - np.array(cv)))])

            x4 = torch.tensor(node_4d, dtype=torch.float32, device=device)
            x6 = torch.tensor(node_6d, dtype=torch.float32, device=device)
            e_idx = torch.tensor([src, dst], dtype=torch.long, device=device)
            e_attr = torch.tensor(attrs, dtype=torch.float32, device=device)
            e_par = torch.tensor(is_par, dtype=torch.bool, device=device)
            s_t = torch.tensor(s_vec, dtype=torch.float32, device=device).unsqueeze(-1)
            target = torch.tensor([[flips[i, 0]]], dtype=torch.float32, device=device)

            for name, m in models.items():
                opt[name].zero_grad()
                if name == "Lange MPNN":
                    pred = m(x4, e_idx)
                elif name == "Neural BP":
                    pred = m(s_t)
                elif name == "ST-GNN":
                    pred = m(x4, e_idx)
                elif name == "Topo-DephaseGNN":
                    pred = m(x6, e_idx, e_attr, e_par)
                loss = crit(pred, target)
                loss.backward()
                opt[name].step()

    for m in models.values():
        m.eval()
    print("[+] Training complete.\n")

def run_diagnostic(shots=30000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"DECODER PAIRWISE AGREEMENT & ERROR DIAGNOSTIC (Shots={shots:,})")
    print("=" * 80 + "\n")

    circuit = make_biased_surface_code(d=5, rounds=5, p_total=0.002, eta=100.0)
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()
    det_coords = circuit.get_detector_coordinates()
    num_dets = circuit.num_detectors

    models = {
        "Lange MPNN": LangeIsotropicMPNN().to(device),
        "Neural BP": NeuralBP().to(device),
        "ST-GNN": STGNN().to(device),
        "Topo-DephaseGNN": TopoDephaseGNN().to(device)
    }

    train_balanced(models, circuit, device, steps=300)

    syn, flips = sampler.sample(shots=shots, separate_observables=True)
    flips = flips.flatten()

    # Base MWPM
    mwpm_preds = matcher.decode_batch(syn).flatten()

    all_preds = {
        "MWPM": mwpm_preds,
        "Lange MPNN": mwpm_preds.copy(),
        "Neural BP": mwpm_preds.copy(),
        "ST-GNN": mwpm_preds.copy(),
        "Topo-DephaseGNN": mwpm_preds.copy()
    }

    complex_idx = np.where(np.sum(syn, axis=1) >= 2)[0]
    print(f"Total Shots: {shots:,} | Multi-Defect Complex Clusters: {len(complex_idx):,} ({len(complex_idx)/shots*100:.2f}%)\n")

    for idx in complex_idx:
        s_vec = syn[idx]
        active = np.where(s_vec)[0]

        node_4d = np.zeros((num_dets, 4), dtype=np.float32)
        node_6d = np.zeros((num_dets, 6), dtype=np.float32)
        for d_idx in range(num_dets):
            c = det_coords.get(d_idx, [0, 0, 0])
            node_4d[d_idx, 0] = s_vec[d_idx]
            node_4d[d_idx, 1:4] = c[:3]
            node_6d[d_idx, 0] = s_vec[d_idx]
            node_6d[d_idx, 1:4] = c[:3]
            node_6d[d_idx, 4] = c[0]
            node_6d[d_idx, 5] = c[1]

        src, dst, is_par, attrs = [], [], [], []
        for a in range(len(active) - 1):
            u, v = active[a], active[a+1]
            cu, cv = det_coords.get(u, [0,0,0]), det_coords.get(v, [0,0,0])
            src.append(u); dst.append(v)
            is_par.append(abs(cu[1] - cv[1]) > abs(cu[0] - cv[0]))
            attrs.append([float(np.linalg.norm(np.array(cu) - np.array(cv)))])

        x4 = torch.tensor(node_4d, dtype=torch.float32, device=device)
        x6 = torch.tensor(node_6d, dtype=torch.float32, device=device)
        e_idx = torch.tensor([src, dst], dtype=torch.long, device=device)
        e_attr = torch.tensor(attrs, dtype=torch.float32, device=device)
        e_par = torch.tensor(is_par, dtype=torch.bool, device=device)
        s_t = torch.tensor(s_vec, dtype=torch.float32, device=device).unsqueeze(-1)

        with torch.no_grad():
            all_preds["Lange MPNN"][idx] = int(models["Lange MPNN"](x4, e_idx).item() > 0.5)
            all_preds["Neural BP"][idx] = int(models["Neural BP"](s_t).item() > 0.5)
            all_preds["ST-GNN"][idx] = int(models["ST-GNN"](x4, e_idx).item() > 0.5)
            
            p_topo = models["Topo-DephaseGNN"](x6, e_idx, e_attr, e_par).item()
            all_preds["Topo-DephaseGNN"][idx] = int(p_topo > 0.5)

    # 1. Error Rate Table
    print(f"{'Decoder':<20} | {'Logical Error Rate':<18} | {'Errors (k)':<10}")
    print("-" * 55)
    for name, p_vec in all_preds.items():
        k = int(np.sum(p_vec != flips))
        p_hat, l, u = wilson_score_interval(k, shots)
        print(f"{name:<20} | {p_hat*100:6.3f}% [{l*100:.2f}%, {u*100:.2f}%] | {k:>4d}/{shots}")
    print("=" * 55 + "\n")

    # 2. Pairwise Prediction Agreement Matrix
    print("Pairwise Prediction Agreement (% of identical predictions across 30,000 shots):")
    names = list(all_preds.keys())
    print(f"{'':<18}" + "".join([f"{n[:10]:>12}" for n in names]))
    for n1 in names:
        row = [f"{np.mean(all_preds[n1] == all_preds[n2])*100:11.2f}%" for n2 in names]
        print(f"{n1:<18}" + "".join(row))

if __name__ == "__main__":
    run_diagnostic()
