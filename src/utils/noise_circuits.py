import stim

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
