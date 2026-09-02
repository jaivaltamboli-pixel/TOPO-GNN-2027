import sinter
import matplotlib.pyplot as plt
import json

def fix_and_plot():
    csv_file = "gate4_benchmark_results.csv"
    print("[+] Parsing and repairing CSV metadata formatting...")
    
    # Read raw lines and fix JSON formatting
    with open(csv_file, "r") as f:
        lines = f.readlines()
        
    header = lines[0]
    fixed_lines = [header]
    
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) >= 7:
            # Rebuild metadata explicitly using valid double-quote JSON syntax
            meta_raw = parts[-1]
            # If it missed proper formatting, parse fields or force clean json string
            # Let's safely extract fields from the row structure
            shots, errors, discards, seconds, decoder, strong_id = parts[:6]
            
            # Reconstruct clean metadata JSON
            # Infer d and p from standard grid if needed, or fix quotes
            fixed_line = f"{shots},{errors},{discards},{seconds},{decoder},{strong_id},{meta_raw.replace("'", '"')}\n"
            fixed_lines.append(fixed_line)
            
    with open(csv_file, "w") as f:
        f.writelines(fixed_lines)

    print("[+] Generating formal Sinter threshold plot...")
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
    fix_and_plot()
