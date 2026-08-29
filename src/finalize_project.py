import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import json
import time
import torch
import numpy as np
import stim
import pymatching
import matplotlib.pyplot as plt

from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from utils.metrics import wilson_score_interval
from models import TopoDephaseGNN, LangeIsotropicMPNN, SpatioTemporalGNN, NeuralBeliefPropagation

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

def generate_final_deliverables(test_distances=[9, 11, 13], p_val=0.002, eta=100.0, shots=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print(f"FINAL PROJECT VALIDATION & BENCHMARK SUITE (Distances: {test_distances}, Shots: {shots:,}, Noise p={p_val}, Bias eta={eta})")
    print("=" * 115 + "\n")

    models = {
        "Topo-DephaseGNN (Proposed)": TopoDephaseGNN().to(device),
        "Lange-inspired MPNN": LangeIsotropicMPNN().to(device),
        "ST-GNN-inspired Net": SpatioTemporalGNN().to(device),
        "Neural BP-inspired Net": NeuralBeliefPropagation().to(device),
    }

    models["Topo-DephaseGNN (Proposed)"].load_state_dict(torch.load("checkpoints/topo_dephase_gnn.pt", map_location=device))
    models["Lange-inspired MPNN"].load_state_dict(torch.load("checkpoints/lange_mpnn.pt", map_location=device))
    models["ST-GNN-inspired Net"].load_state_dict(torch.load("checkpoints/st_gnn.pt", map_location=device))
    models["Neural BP-inspired Net"].load_state_dict(torch.load("checkpoints/neural_bp.pt", map_location=device))

    for m in models.values():
        m.eval()

    benchmark_summary = {
        "metadata": {
            "test_distances": test_distances,
            "physical_error_rate": p_val,
            "bias_eta": eta,
            "shots_per_distance": shots
        },
        "results": {}
    }

    for d in test_distances:
        t_dist_start = time.time()
        circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
        dem = circuit.detector_error_model(decompose_errors=True)
        coords = circuit.get_detector_coordinates()
        num_dets = circuit.num_detectors
        edge_dict, bnd_z_idx, bnd_x_idx, _ = extract_complete_dem_graph(dem, num_dets, coords, d)

        sampler = circuit.compile_detector_sampler()
        syn, flips = sampler.sample(shots=shots, separate_observables=True)
        flips = flips.flatten().astype(np.int64)

        # Baseline MWPM
        t0 = time.time()
        base_matcher = pymatching.Matching.from_detector_error_model(dem)
        mwpm_preds = base_matcher.decode_batch(syn).flatten().astype(np.int64)
        mwpm_lat = (time.time() - t0) * 1000.0 / (shots / 1000.0)

        preds = {
            "MWPM (Classical Baseline)": mwpm_preds,
            "Topo-DephaseGNN (Proposed)": np.zeros(shots, dtype=np.int64),
            "Lange-inspired MPNN": np.zeros(shots, dtype=np.int64),
            "ST-GNN-inspired Net": np.zeros(shots, dtype=np.int64),
            "Neural BP-inspired Net": np.zeros(shots, dtype=np.int64),
        }

        active_shots = np.where(np.sum(syn, axis=1) >= 2)[0]
        t_neural_start = time.time()

        for idx in active_shots:
            s = syn[idx].astype(np.uint8)
            x4, x6, e_idx, e_attr, e_par, s_t, _, _ = extract_active_subgraph_tensors(
                s, coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device
            )
            if e_idx.numel() == 0:
                continue

            with torch.no_grad():
                preds["Lange-inspired MPNN"][idx] = int(models["Lange-inspired MPNN"](x4, e_idx, e_attr).item() > 0.5)
                preds["Neural BP-inspired Net"][idx] = int(models["Neural BP-inspired Net"](s_t, e_idx, e_attr).item() > 0.5)
                preds["ST-GNN-inspired Net"][idx] = int(models["ST-GNN-inspired Net"](x4, e_idx, e_attr).item() > 0.5)
                log_pred, _ = models["Topo-DephaseGNN (Proposed)"](x6, e_idx, e_attr, e_par)
                preds["Topo-DephaseGNN (Proposed)"][idx] = int(log_pred.item() > 0.5)

        neural_lat = (time.time() - t_neural_start) * 1000.0 / (shots / 1000.0) / 4.0

        benchmark_summary["results"][str(d)] = {}
        print(f"============================== ZERO-SHOT EVALUATION: DISTANCE d = {d} ==============================")
        print(f"{'Decoder / Architecture':<36} | {'Logical Error':<14} | {'95% Wilson CI':<18} | {'Errors':<10} | {'Latency (ms/1k)':<15}")
        print("-" * 115)

        for name, p_vec in preds.items():
            k_err = int(np.sum(p_vec != flips))
            p_hat, l, u = wilson_score_interval(k_err, shots)
            lat = mwpm_lat if "MWPM" in name else neural_lat
            benchmark_summary["results"][str(d)][name] = {
                "logical_error": float(p_hat),
                "ci_lower": float(l),
                "ci_upper": float(u),
                "errors": k_err,
                "shots": shots,
                "latency_ms": float(lat)
            }
            print(f"{name:<36} | {p_hat * 100:6.3f}%       | [{l * 100:.3f}%, {u * 100:.3f}%]   | {k_err:>5d}/{shots} | {lat:6.2f} ms")

        print("-" * 115)
        print(f"  [Timing] Distance d={d} finished in {time.time() - t_dist_start:.2f}s\n")

    with open("results/final_benchmark.json", "w") as f:
        json.dump(benchmark_summary, f, indent=2)
    print("  [+] Saved benchmark metrics to results/final_benchmark.json")

    # Generate Final Scaling Figure
    plt.figure(figsize=(8.5, 6))
    palette = {
        "MWPM (Classical Baseline)": ("#1f77b4", "o", "-"),
        "Topo-DephaseGNN (Proposed)": ("#2ca02c", "s", "--"),
        "Lange-inspired MPNN": ("#ff7f0e", "^", ":"),
        "ST-GNN-inspired Net": ("#9467bd", "v", ":"),
        "Neural BP-inspired Net": ("#d62728", "x", "-.")
    }

    for name, (col, mark, lstyle) in palette.items():
        errs = np.array([benchmark_summary["results"][str(d)][name]["logical_error"] for d in test_distances])
        lows = np.array([benchmark_summary["results"][str(d)][name]["ci_lower"] for d in test_distances])
        highs = np.array([benchmark_summary["results"][str(d)][name]["ci_upper"] for d in test_distances])
        
        lower_err = np.clip(errs - lows, a_min=0.0, a_max=None)
        upper_err = np.clip(highs - errs, a_min=0.0, a_max=None)
        yerr = [lower_err, upper_err]

        plt.errorbar(test_distances, errs, yerr=yerr, label=name, color=col, marker=mark, linestyle=lstyle, linewidth=2, capsize=4)

    plt.xlabel("Surface Code Distance ($d$)", fontsize=11, fontweight="bold")
    plt.ylabel("Logical Error Rate ($P_L$)", fontsize=11, fontweight="bold")
    plt.title(r"Zero-Shot Scaling Across Unseen Distances under Biased Noise ($\eta=100, p=0.002$)", fontsize=12, fontweight="bold")
    plt.xticks(test_distances)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, loc="upper left")
    plt.tight_layout()

    plt.savefig("figures/final_scaling_curves.png", dpi=300)
    plt.close()
    print("  [+] Saved publication scaling curves to figures/final_scaling_curves.png\n")

if __name__ == "__main__":
    generate_final_deliverables()
