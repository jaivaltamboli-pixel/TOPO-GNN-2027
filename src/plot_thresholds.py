import sinter
import matplotlib.pyplot as plt

def render_plot():
    csv_file = "gate4_benchmark_results.csv"
    print("[+] Reading stats and rendering threshold plot with updated Sinter API...")
    
    stats = sinter.read_stats_from_csv_files(csv_file)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda stat: stat.json_metadata["p"],
        group_func=lambda stat: f"d={stat.json_metadata['d']} ({stat.decoder})",
        failure_units_per_shot_func=lambda stat: stat.json_metadata["d"]
    )
    
    ax.loglog()
    ax.set_title(r"Gate 4: Topo-GNN vs MWPM Threshold Benchmark (Biased Noise $\eta=100$)")
    ax.set_xlabel("Physical Error Rate (p)")
    ax.set_ylabel("Logical Error Rate per Round ($P_L$)")
    ax.grid(which="major", color="black", alpha=0.3)
    ax.grid(which="minor", color="gray", alpha=0.1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plot_path = "images/gate4_threshold_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"[+] Gate 4 Official Plot successfully saved to: {plot_path}")

if __name__ == "__main__":
    render_plot()
