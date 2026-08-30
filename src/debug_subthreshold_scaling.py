import os
os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = ""

import numpy as np
import stim
import pymatching
from utils.noise_circuits import make_biased_surface_code

def check_subthreshold_baseline():
    distances = [3, 5, 7, 9]
    p_sub = [0.0005, 0.001, 0.002]
    shots = 10000
    eta = 100.0

    print("=" * 90)
    print(f"SUB-THRESHOLD MWPM BASELINE AUDIT ({shots:,} shots/point, Bias eta={eta})")
    print("=" * 90 + "\n")

    print(f"{'p_phys':<10} | {'d=3 LER':<12} | {'d=5 LER':<12} | {'d=7 LER':<12} | {'d=9 LER':<12} | {'Scaling Status'}")
    print("-" * 90)

    for p in p_sub:
        lers = []
        for d in distances:
            circuit = make_biased_surface_code(d=d, rounds=d, p_total=p, eta=eta)
            dem = circuit.detector_error_model(decompose_errors=True)
            matcher = pymatching.Matching.from_detector_error_model(dem)
            sampler = circuit.compile_detector_sampler()

            syn, flips = sampler.sample(shots=shots, separate_observables=True)
            preds = matcher.decode_batch(syn).flatten().astype(np.int64)
            flips = flips.flatten().astype(np.int64)

            err = np.mean(preds != flips) * 100.0
            lers.append(err)

        is_suppressing = all(lers[i] > lers[i+1] for i in range(len(lers)-1))
        status = "EXPONENTIAL SUPPRESSION (Sub-threshold)" if is_suppressing else "NO SUPPRESSION (Above threshold)"
        print(f"p={p:<8.4f} | {lers[0]:6.3f}%     | {lers[1]:6.3f}%     | {lers[2]:6.3f}%     | {lers[3]:6.3f}%     | {status}")

    print("=" * 90 + "\n")

if __name__ == "__main__":
    check_subthreshold_baseline()
