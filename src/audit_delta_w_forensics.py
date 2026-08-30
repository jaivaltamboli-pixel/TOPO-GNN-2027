import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import time
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from audit_phase_a_d_opportunity import (
    build_parity_expanded_graph,
    find_exact_logical_reference_chain,
    standardize_edge,
    compute_chain_observable,
    compute_chain_weight
)

def run_delta_w_forensics(distances=[3, 5, 7, 9], p_val=0.002, eta=100.0, shots=100000):
    print("=" * 120)
    print(f"DIAGNOSTIC 2: CLASSICAL WEIGHT GAP FORENSICS (DELTA W = W_B - W_A) ({shots:,} shots/distance, p={p_val})")
    print("=" * 120 + "\n")

    for d in distances:
        t0 = time.time()
        circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
        dem = circuit.detector_error_model(decompose_errors=True)
        coords = circuit.get_detector_coordinates()
        num_dets = circuit.num_detectors

        edge_dict, adj, bnd_z_idx, bnd_x_idx = build_parity_expanded_graph(dem, num_dets, coords, d)
        R_L, w_ref = find_exact_logical_reference_chain(adj, num_dets)

        matcher = pymatching.Matching.from_detector_error_model(dem)
        sampler = circuit.compile_detector_sampler()
        syn, flips = sampler.sample(shots=shots, separate_observables=True)
        flips = flips.flatten().astype(np.int64)

        gaps_on_failures = []
        gaps_on_correct = []
        higher_weight_failures = 0
        lower_weight_failures = 0

        for idx in range(shots):
            s = syn[idx].astype(np.uint8)
            y_true = flips[idx]

            edges_a_raw = matcher.decode_to_edges_array(s)
            C_A = set(standardize_edge(int(e[0]), int(e[1]), bnd_z_idx, bnd_x_idx) for e in edges_a_raw)
            C_B = C_A.symmetric_difference(R_L)

            obs_A = compute_chain_observable(C_A, edge_dict)
            w_A = compute_chain_weight(C_A, edge_dict)
            w_B = compute_chain_weight(C_B, edge_dict)
            delta_w = w_B - w_A  # Always >= 0 since C_A is Blossom's global minimum

            if obs_A != y_true:
                gaps_on_failures.append(delta_w)
                # Correct candidate is C_B, which has classical weight W_B >= W_A
                if w_B > w_A:
                    higher_weight_failures += 1
                elif w_B < w_A:
                    lower_weight_failures += 1
            else:
                gaps_on_correct.append(delta_w)

        err_arr = np.array(gaps_on_failures)
        corr_arr = np.array(gaps_on_correct)
        n_fail = len(err_arr)
        n_corr = len(corr_arr)

        print(f"============================== DISTANCE d = {d:2d} ({time.time()-t0:5.2f}s) ==============================")
        print(f"  MWPM Logical Error Rate:                     {n_fail:>5d}/{shots:,} ({n_fail/shots*100:6.3f}%)")
        print(f"  Recoverable Failures (C_B is True):          {n_fail:>5d}/{n_fail:>5d} (100.00%)")
        print(f"  Fraction of Failures where W_correct > W_MWPM: {higher_weight_failures:>5d}/{n_fail:>5d} ({higher_weight_failures/n_fail*100:6.2f}%)")
        print(f"  Fraction of Failures where W_correct < W_MWPM: {lower_weight_failures:>5d}/{n_fail:>5d} ({lower_weight_failures/n_fail*100:6.2f}%)")
        print()
        print(f"  [MWPM FAILURES] (The target population to overturn):")
        print(f"    Mean Delta W:   {np.mean(err_arr):6.3f}")
        print(f"    Median Delta W: {np.median(err_arr):6.3f}")
        print(f"    Min / Max:      {np.min(err_arr):6.3f} / {np.max(err_arr):6.3f}")
        print(f"    Percentiles [10%, 25%, 50%, 75%, 90%]:")
        print(f"      [{np.percentile(err_arr, 10):.3f}, {np.percentile(err_arr, 25):.3f}, {np.percentile(err_arr, 50):.3f}, {np.percentile(err_arr, 75):.3f}, {np.percentile(err_arr, 90):.3f}]")
        print()
        print(f"  [MWPM CORRECT SHOTS] (The baseline population to protect):")
        print(f"    Mean Delta W:   {np.mean(corr_arr):6.3f}")
        print(f"    Median Delta W: {np.median(corr_arr):6.3f}")
        print(f"    Min / Max:      {np.min(corr_arr):6.3f} / {np.max(corr_arr):6.3f}")
        print(f"    Percentiles [10%, 25%, 50%, 75%, 90%]:")
        print(f"      [{np.percentile(corr_arr, 10):.3f}, {np.percentile(corr_arr, 25):.3f}, {np.percentile(corr_arr, 50):.3f}, {np.percentile(corr_arr, 75):.3f}, {np.percentile(corr_arr, 90):.3f}]")
        print("-" * 120 + "\n")

if __name__ == "__main__":
    run_delta_w_forensics()
