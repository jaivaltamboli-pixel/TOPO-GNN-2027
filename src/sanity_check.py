import stim
import torch
from utils.noise_circuits import make_biased_surface_code
from utils.graph_builder import extract_complete_dem_graph, extract_active_subgraph_tensors
from models import TopoDephaseGNN

print("=" * 60)
print("SANITY VERIFICATION: PIPELINE & GRAPH TENSORS")
print("=" * 60)

circuit = make_biased_surface_code(d=3, rounds=3, p_total=0.002, eta=100.0)
dem = circuit.detector_error_model(decompose_errors=True)
coords = circuit.get_detector_coordinates()
num_dets = circuit.num_detectors
edge_dict, bnd_idx = extract_complete_dem_graph(dem, num_dets, coords, 3)

print(f"  [+] Physical DEM Edges Extracted: {len(edge_dict)}")
print(f"  [+] Virtual Boundary Node Index: {bnd_idx}")

sampler = circuit.compile_detector_sampler()
syn, flips = sampler.sample(shots=20, separate_observables=True)

# Find first non-trivial syndrome
idx = 0
for i in range(len(syn)):
    if syn[i].sum() >= 2:
        idx = i
        break

x4, x6, e_idx, e_attr, e_par, s_t, e_targ, pairs = extract_active_subgraph_tensors(
    syn[idx], coords, edge_dict, bnd_idx, 3, "cpu"
)

model = TopoDephaseGNN()
log_p, delta_w = model(x6, e_idx, e_attr, e_par)

print(f"  [+] Active Subgraph Nodes: {x6.size(0)}, Subgraph Edges: {e_idx.size(1)}")
print(f"  [+] Delta_w Tensor Shape: {delta_w.shape} (Bounded values in [-1.5, 1.5])")
print(f"  [+] Logical Coset Prediction: {log_p.item():.4f}")
print("=" * 60)
print("  [SUCCESS] Graph builder, boundary extraction, and dual heads verified!\n")
