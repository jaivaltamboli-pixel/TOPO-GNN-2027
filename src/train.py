import torch
import torch.nn as nn
import numpy as np
import json
import time

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import LangeIsotropicMPNN, NeuralBeliefPropagation, SpatioTemporalGNN, TopoDephaseGNN

def run_training(p_val=0.002, eta=100.0, train_steps_per_d=180):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print(f"TRAINING PHASE: Multi-Distance Physical Edge & Coset Optimization ({device})")
    print("=" * 90 + "\n")

    models = {
        "lange_mpnn": LangeIsotropicMPNN().to(device),
        "neural_bp": NeuralBeliefPropagation().to(device),
        "st_gnn": SpatioTemporalGNN().to(device),
        "topo_dephase_gnn": TopoDephaseGNN().to(device)
    }

    opt = {k: torch.optim.AdamW(v.parameters(), lr=8e-4, weight_decay=1e-4) for k, v in models.items()}
    bce_cls = nn.BCELoss()
    bce_edge = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.5], device=device))

    for d in [3, 5, 7]:
        t0 = time.time()
        circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
        dem = circuit.detector_error_model(decompose_errors=True)
        coords = circuit.get_detector_coordinates()
        num_dets = circuit.num_detectors
        edge_dict, bnd_z_idx, bnd_x_idx, dem_fault_to_edge = extract_complete_dem_graph(dem, num_dets, coords, d)
        dem_sampler = dem.compile_sampler()

        for step in range(train_steps_per_d):
            det_data, obs_data, err_data = dem_sampler.sample(shots=64, return_errors=True)
            for i in range(len(det_data)):
                s = det_data[i]
                if np.sum(s) < 1:
                    continue

                active_fault_indices = np.where(err_data[i])[0]
                active_fault_pairs = set()
                for f_idx in active_fault_indices:
                    if f_idx < len(dem_fault_to_edge):
                        pair = dem_fault_to_edge[f_idx]
                        if pair is not None:
                            active_fault_pairs.add(pair)

                x4, x6, e_idx, e_attr, e_par, s_t, e_targ, _ = extract_active_subgraph_tensors(
                    s, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device, active_fault_pairs=active_fault_pairs
                )
                if e_idx.size(1) == 0:
                    continue

                target_cls = torch.tensor([[obs_data[i, 0]]], dtype=torch.float32, device=device)

                for name, m in models.items():
                    opt[name].zero_grad()
                    if name == "lange_mpnn":
                        loss = bce_cls(m(x4, e_idx, e_attr), target_cls)
                    elif name == "neural_bp":
                        loss = bce_cls(m(s_t, e_idx, e_attr), target_cls)
                    elif name == "st_gnn":
                        loss = bce_cls(m(x4, e_idx, e_attr), target_cls)
                    elif name == "topo_dephase_gnn":
                        log_pred, edge_logits = m(x6, e_idx, e_attr, e_par)
                        loss_c = bce_cls(log_pred, target_cls)
                        loss_e = bce_edge(edge_logits, e_targ) if edge_logits.numel() > 0 else 0.0
                        loss = loss_c + 0.6 * loss_e
                        
                    loss.backward()
                    opt[name].step()

        print(f"  [+] Finished training on d={d} ({time.time()-t0:.2f}s)")

    for name, m in models.items():
        torch.save(m.state_dict(), f"checkpoints/{name}.pt")

    meta = {"training_distances": [3, 5, 7], "p_val": p_val, "eta": eta}
    with open("checkpoints/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("  [*] Model checkpoints locked in checkpoints/\n")

if __name__ == "__main__":
    run_training()
