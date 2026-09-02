import sinter
import matplotlib.pyplot as plt
import json

def clean_and_plot():
    csv_file = "gate4_benchmark_results.csv"
    print("[+] Rebuilding CSV with pristine Sinter JSON metadata...")
    
    with open(csv_file, "r") as f:
        lines = f.readlines()
        
    fixed_rows = ["shots,errors,discards,seconds,decoder,strong_id,json_metadata\n"]
    
    # We know the exact grid we ran: d in [3, 5, 7, 9] and p_val in error_rates
    distances = [3, 5, 7, 9]
    error_rates = [0.001, 0.002, 0.0035, 0.005, 0.008]
    eta = 100.0
    shots = 50000
    
    # Re-parse original data lines by index alignment
    # Each distance * error rate pair produced 2 lines (mwpm, topo_gnn)
    line_idx = 1
    for d in distances:
        for p in error_rates:
            if line_idx < len(lines):
                parts = lines[line_idx].strip().split(",")
                mwpm_errs = parts[1]
                t_run = parts[3]
                line_idx += 1
            if line_idx < len(lines):
                parts = lines[line_idx].strip().split(",")
                topo_errs = parts[1]
                line_idx += 1
                
            # Valid JSON metadata using standard double quotes and json.dumps
            meta_mwpm = json.dumps({"d": d, "p": p, "eta": eta})
            meta_topo = json.dumps({"d": d, "p": p, "eta": eta})
            
            fixed_rows.append(f"{shots},{mwpm_errs},0,{t_run},mwpm,_,\"{meta_mwpm}\"\n")
            fixed_rows.append(f"{shots},{topo_errs},0,{t_run},topo_gnn,_,\"{meta_topo}\"\n")
            
    with open(csv_file, "w") as f:
        f.writelines(fixed_rows)

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
    clean_and_plot()
