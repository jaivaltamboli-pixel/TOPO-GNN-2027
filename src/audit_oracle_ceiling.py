import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import time
import numpy as np
import stim
import pymatching

from utils.noise_circuits import make_biased_surface_code
from utils.metrics import wilson_score_interval
from audit_phase_a_d_opportunity import (
    build_parity_expanded_graph,
    find_exact_logical_reference_chain,
    standardize_edge,
    compute_chain_boundary,
    compute_chain_observable
)

def run_oracle_dual_coset_ceiling_audit(
    distances=[3, 5, 7, 9],
    p_val=0.002,
    eta=100.0,
    shots=100000
):
    print("=" * 120)
    print(f"ORACLE DUAL-COSET THEORETICAL CEILING AUDIT ({shots:,} shots/distance, p={p_val}, Bias eta={eta})")
    print("=" * 120 + "\n")

    summary_rows = []

    for d in distances:
        t_start = time.time()
        
        # 1. Circuit, DEM, and Graph extraction
        circuit = make_biased_surface_code(d=d, rounds=d, p_total=p_val, eta=eta)
        dem = circuit.detector_error_model(decompose_errors=True)
        coords = circuit.get_detector_coordinates()
        num_dets = circuit.num_detectors

        edge_dict, adj, bnd_z_idx, bnd_x_idx = build_parity_expanded_graph(dem, num_dets, coords, d)
        
        # 2. Extract validated R_L chain
        R_L, w_ref = find_exact_logical_reference_chain(adj, num_dets)
        
        # 3. Reference chain invariant checks
        bnd_R_L = compute_chain_boundary(R_L, num_dets)
        obs_R_L = compute_chain_observable(R_L, edge_dict)
        if not np.all(bnd_R_L == 0):
            raise AssertionError(f"FATAL: d={d} reference chain R_L has non-zero detector boundary: {np.where(bnd_R_L > 0)[0]}")
        if obs_R_L != 1:
            raise AssertionError(f"FATAL: d={d} reference chain R_L does not have odd observable parity (obs={obs_R_L})!")

        # 4. Initialize Matcher & Sampler ONCE per distance
        matcher = pymatching.Matching.from_detector_error_model(dem)
        sampler = circuit.compile_detector_sampler()
        syn, flips = sampler.sample(shots=shots, separate_observables=True)
        flips = flips.flatten().astype(np.int64)

        mwpm_errors = 0
        oracle_errors = 0
        recoverable_failures = 0
        oracle_overturns = 0

        for idx in range(shots):
            s = syn[idx].astype(np.uint8)
            y_true = flips[idx]

            # Primary Chain C_A from Blossom
            edges_a_raw = matcher.decode_to_edges_array(s)
            C_A = set(standardize_edge(int(e[0]), int(e[1]), bnd_z_idx, bnd_x_idx) for e in edges_a_raw)

            # Exact Complementary Chain C_B = C_A XOR R_L
            C_B = C_A.symmetric_difference(R_L)

            # Strict Invariant Verification per shot
            bnd_A = compute_chain_boundary(C_A, num_dets)
            bnd_B = compute_chain_boundary(C_B, num_dets)
            obs_A = compute_chain_observable(C_A, edge_dict)
            obs_B = compute_chain_observable(C_B, edge_dict)

            if not np.array_equal(bnd_A, s):
                raise AssertionError(f"FATAL: Shot {idx} at d={d}: partial(C_A) != syndrome!")
            if not np.array_equal(bnd_B, s):
                raise AssertionError(f"FATAL: Shot {idx} at d={d}: partial(C_B) != syndrome!")
            if obs_A == obs_B:
                raise AssertionError(f"FATAL: Shot {idx} at d={d}: Homology parity failure (obs_A == obs_B == {obs_A})!")

            # Evaluate Classical MWPM
            if obs_A != y_true:
                mwpm_errors += 1
                if obs_B == y_true:
                    recoverable_failures += 1

            # Evaluate Oracle Selector
            # Oracle chooses the correct candidate if either C_A or C_B matches y_true
            if obs_A == y_true or obs_B == y_true:
                chosen_candidate_obs = y_true
            else:
                # Both candidates failed (neither is correct)
                chosen_candidate_obs = obs_A
                oracle_errors += 1

            if chosen_candidate_obs != obs_A:
                oracle_overturns += 1

        elapsed = time.time() - t_start

        # Wilson Confidence Intervals
        mwpm_p_hat, mwpm_ci_l, mwpm_ci_u = wilson_score_interval(mwpm_errors, shots)
        oracle_p_hat, oracle_ci_l, oracle_ci_u = wilson_score_interval(oracle_errors, shots)

        rec_rate = (recoverable_failures / mwpm_errors * 100.0) if mwpm_errors > 0 else 0.0

        summary_rows.append({
            "d": d,
            "mwpm_errs": mwpm_errors,
            "mwpm_p_hat": mwpm_p_hat,
            "mwpm_ci": (mwpm_ci_l, mwpm_ci_u),
            "oracle_errs": oracle_errors,
            "oracle_p_hat": oracle_p_hat,
            "oracle_ci": (oracle_ci_l, oracle_ci_u),
            "recoverable": recoverable_failures,
            "rec_rate": rec_rate,
            "overturns": oracle_overturns,
            "time": elapsed
        })

        print(f"============================== DISTANCE d = {d:2d} ({elapsed:5.2f}s) ==============================")
        print(f"  MWPM Logical Errors:          {mwpm_errors:>6d}/{shots:,} ({mwpm_p_hat*100:6.3f}%) | 95% CI: [{mwpm_ci_l*100:.3f}%, {mwpm_ci_u*100:.3f}%]")
        print(f"  Oracle Logical Errors:        {oracle_errors:>6d}/{shots:,} ({oracle_p_hat*100:6.3f}%) | 95% CI: [{oracle_ci_l*100:.3f}%, {oracle_ci_u*100:.3f}%]")
        print(f"  MWPM Failures Recoverable:    {recoverable_failures:>6d}/{mwpm_errors:>6d} ({rec_rate:6.2f}%)")
        print(f"  Oracle Altered Decisions:     {oracle_overturns:>6d}/{shots:,}")
        print("-" * 120 + "\n")

    # Output Concise Summary Table
    print("=" * 120)
    print("FINAL CONCISE ORACLE THEORETICAL CEILING TABLE")
    print("=" * 120)
    print(f"{'Distance':<10} | {'MWPM P_L (95% CI)':<26} | {'Oracle P_L (95% CI)':<26} | {'MWPM Errs':<10} | {'Recoverable':<12} | {'Recoverability':<15} | {'Oracle Improvement':<18}")
    print("-" * 120)
    for row in summary_rows:
        mwpm_str = f"{row['mwpm_p_hat']*100:5.3f}% [{row['mwpm_ci'][0]*100:.3f}%, {row['mwpm_ci'][1]*100:.3f}%]"
        oracle_str = f"{row['oracle_p_hat']*100:5.3f}% [{row['oracle_ci'][0]*100:.3f}%, {row['oracle_ci'][1]*100:.3f}%]"
        impr_str = f"{(row['mwpm_p_hat'] - row['oracle_p_hat'])*100:5.3f}% (-{row['rec_rate']:.1f}%)"
        print(f"d = {row['d']:<6d} | {mwpm_str:<26} | {oracle_str:<26} | {row['mwpm_errs']:>5d}/{shots} | {row['recoverable']:>5d}/{row['mwpm_errs']} | {row['rec_rate']:6.2f}%        | {impr_str:<18}")
    print("=" * 120 + "\n")

if __name__ == "__main__":
    run_oracle_dual_coset_ceiling_audit()
