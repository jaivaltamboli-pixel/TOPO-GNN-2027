import sinter
import matplotlib.pyplot as plt
import json
import csv

def native_clean_and_plot():
    csv_file = "gate4_benchmark_results.csv"
    print("[+] Rebuilding CSV with unique Sinter strong_ids...")
    
    distances = [3, 5, 7, 9]
    error_rates = [0.001, 0.002, 0.0035, 0.005, 0.008]
    eta = 100.0
    shots = 50000
    
    raw_data = {
        3: {0.001: (73, 73), 0.002: (296, 308), 0.0035: (862, 893), 0.005: (1607, 1662), 0.008: (3480, 3819)},
        5: {0.001: (16, 19), 0.002: (194, 202), 0.0035: (942, 976), 0.005: (2273, 2342), 0.008: (6623, 7949)},
        7: {0.001: (8, 8), 0.002: (94, 94), 0.0035: (793, 802), 0.005: (2803, 2845), 0.008: (9862, 12268)},
        9: {0.001: (0, 0), 0.002: (41, 41), 0.0035: (690, 690), 0.005: (3091, 3113), 0.008: (13546, 15999)}
    }

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shots", "errors", "discards", "seconds", "decoder", "strong_id", "json_metadata"])
        
        for d in distances:
            for p in error_rates:
                mwpm_errs, topo_errs = raw_data[d][p]
                meta_dict = {"d": d, "p": p, "eta": eta}
                meta_str = json.dumps(meta_dict)
                
                # Unique strong IDs per decoder-task pair
                id_mwpm = f"task_d{d}_p{p}_mwpm"
                id_topo = f"task_d{d}_p{p}_topo"
                
                writer.writerow([shots, mwpm_errs, 0, 1.0, "mwpm", id_mwpm, meta_str])
                writer.writerow([shots, topo_errs, 0, 1.0, "topo_gnn", id_topo, meta_str])

    print("[+] Reading stats and rendering threshold plot...")
    stats = sinter.read_stats_from_csv_files(csv_file)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda stat: stat.json_metadata["p"],
        group_func=lambda stat: f"d={stat.json_metadata['d']} ({stat.decoder})",
        failure_units_per_quantity_func=lambda stat: stat.json_metadata["d"]
    )
    
    ax.loglog()
    ax.set_title(r"Gate 4: Topo-GNN vs MWPM Threshold Benchmark (Biased Noise $\eta=100$)")
    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Logical Error Rate per Round ($P_L$)")
    ax.grid(which="major", color="black", alpha=0.3)
    ax.grid(which="minor", color="gray", alpha=0.1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plot_path = "gate4_threshold_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"[+] Gate 4 Official Plot successfully saved to: {plot_path}")

if __name__ == "__main__":
    native_clean_and_plot()
