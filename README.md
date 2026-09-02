# TOPO-GNN-2026: Topological Graph Neural Network Decoder for Quantum Error Correction

A production-grade, hybrid classical-AI quantum error correction (QEC) decoder built from scratch to evaluate and enhance surface code performance under highly biased physical noise ($\eta = 100$). The framework integrates **Stim** for stabilizer circuit generation, **PyMatching** for classical Minimum Weight Perfect Matching (MWPM), a custom **6-layer Relational Message-Passing GNN (`TopoOracle`)** in PyTorch, and the **Sinter** simulation framework for rigorous threshold benchmarking.

---

## Architectural Highlights

* **Relational Message-Passing (`TopoOracle`):** A custom 6-layer PyTorch GNN that ingests lattice coordinate tensors, dynamic syndrome states, and parity-expanded detector error model (DEM) graphs to compute homological cycle energy adjustments.
* **Symmetric Data Filtering:** Implements strict symmetric filtering across training pools to eliminate hidden data leakage, forcing the neural network away from naive classical weight differences and toward true topological learning.
* **Mathematical Safety Guardrail:** Operates as a conservative correction mechanism. By enforcing confidence thresholds ($\tau \ge 0.75$), the model achieves a **Zero Regressions** guarantee—refusing to corrupt correct classical decodings while targeting ambiguous boundary errors.
* **High-Throughput Evaluation Engine:** Features a GPU-accelerated PyTorch evaluation loop designed to bypass standard multiprocessing limitations of the Sinter framework, mapping scaling behavior across distances up to $d=9$.

---

## Project Structure

```text
TOPO-GNN-2027/
├── models/
│   └── topo_gnn_gate4.pt          # Serialized production weights for TopoOracle
├── src/
│   ├── topo_oracle_model.py           # Parity expansion, reference chains, and safety audits
│   ├── train_decoder.py               # Training and weight serialization pipeline
│   ├── benchmark_sinter.py            # Sinter-compatible evaluation and data collection
│   └── plot_thresholds.py             # Official log-log threshold curve rendering script
├── utils/
│   ├── graph_builder.py               # DEM graph extraction utilities
│   └── noise_circuits.py              # Biased surface code circuit generation (Stim)
├── gate4_benchmark_results.csv        # Multi-baseline Sinter performance data
└── images/gate4_threshold_plot.png           # Official log-log threshold comparison chart

```

---

## Milestones & Execution Gates

* **Gate 1 & 2 (Foundation & Pipeline):** Established raw circuit synthesis via Stim, constructed parity-expanded graphs, and integrated PyMatching baseline decoders.
* **Gate 3 (Leakage Audit & Zero-Shot Extrapolation):** Resolved lethal data leakage via symmetric filtering, validated zero regressions at $\tau \ge 0.75$, and mapped the zero-shot extrapolation ceiling at $d=11$.
* **Gate 4 (Production Pipeline & Sinter Benchmarking):** Serialized the trained neural engine into `topo_gnn_gate4.pt`, executed high-throughput multi-distance sweeps ($d \in \{3, 5, 7, 9\}$), and generated official performance logs and threshold visualizations.

---

## Getting Started

### 1. Requirements & Dependencies

Ensure Python 3.10+ and a CUDA-compatible GPU environment are active, along with core QEC libraries:

```bash
pip install stim pymatching torch sinter matplotlib networkx numpy

```

### 2. Train and Save Production Weights

Execute the training loop on combined distance caches ($d=5, 7, 9$) to serialize the core neural engine:

```bash
python src/train_decoder.py

```

### 3. Run the Benchmark & Generate Threshold Plots

Execute the evaluation sweep across noise rates and render the final Sinter comparative analysis:

```bash
python src/benchmark_sinter.py
python src/plot_thresholds.py

```

---

## Results & Findings

* **Zero Regressions Verified:** At strict confidence gates ($\tau \ge 0.75$), the model maintains absolute safety by ensuring zero logical regressions on test evaluations.
* **Scaling Boundary Identification:** Empirical testing demonstrated that a fixed-depth 6-layer GNN reaches receptive field saturation at $d \ge 11$, establishing clear design criteria for future multiscale or hierarchical iterations.
