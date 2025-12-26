```markdown
# The Bottleneck Effect: Why Model Scaling Fails for Code Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Paper Status](https://img.shields.io/badge/Status-Research_Preview-green.svg)]()

> **"Does making an LLM bigger always make it smarter? Our execution-aware analysis says no."**

This repository contains the official implementation and experimental data for the paper **"The Bottleneck Effect: Why Model Scaling Fails for Code Generation"**. unlike prior work that treats code generation as simple text prediction, we analyze failures through the lens of **execution semantics** and **internal activation geometry**.

---

## 📉 Key Findings

After running **16,400+ execution-based experiments** across HumanEval and MBPP, we uncovered two critical anomalies:

### 1. The Negative Scaling Phenomenon
Contrary to standard scaling laws, **GPT-2 Medium (355M)** consistently underperforms **GPT-2 Small (124M)** on syntax-heavy tasks.
- **Small (124M):** Higher pass@1 rate on structural code.
- **Medium (355M):** Frequently collapses into repetitive failure modes.

### 2. The Geometric Bottleneck
By projecting layer-wise activations, we identified a **"Single Point of Failure"** in Layer 12 of the medium model.
- **Linear Separability:** Successful vs. Failed generations are linearly separable along just two dimensions (Dim 810 & 457).
- **The "Rigid" Trap:** A deviation of just **4.5 units** in this bottleneck layer causes catastrophic syntax failure, proving that larger models can form more brittle decision boundaries.

---

## 📁 Repository Structure


```

paper11_bottleneck_effect/
├── data/                 # Raw execution logs and activation dumps
├── src/                  # Core research code
│   ├── models/           # Wrappers for GPT-2/CodeGen with hooks
│   ├── execution/        # Sandbox for running generated code
│   └── analysis/         # PCA & Linear Probe tools (Figure 4)
├── scripts/              # Reproduction scripts
├── notebooks/            # Visualization notebooks (Heatmaps & Scatter plots)
└── outputs/              # Saved figures and failure taxonomy

```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone [https://github.com/yourusername/LLM-Bottleneck-Effect.git](https://github.com/yourusername/LLM-Bottleneck-Effect.git)
cd LLM-Bottleneck-Effect

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

```

### 2. Download Datasets

We use HuggingFace datasets for HumanEval and MBPP.

```bash
python scripts/download_data.py --dataset all

```

---

## 🧪 Reproduction

To replicate the results from the paper, follow these steps:

### Step 1: Run the Generation Loop (The "16k Experiments")

This script generates solutions and immediately executes them to label "Success" vs "Failure".

```bash
python scripts/run_experiment.py \
    --model gpt2-medium \
    --num_samples 100 \
    --output_dir outputs/raw_logs

```

### Step 2: Extract Activations (The "Bottleneck" Analysis)

This hooks into Layer 12 to capture the hidden states during generation.

```bash
python scripts/extract_activations.py \
    --checkpoint gpt2-medium \
    --layer 11 \
    --target_dim 2

```

### Step 3: Visualize the Geometric Trap (Figure 4)

Generate the scatter plot showing the linear separability of failures.

```bash
jupyter notebook notebooks/visualize_bottleneck.ipynb

```

---

## 🔧 Configuration

All experimental parameters are controlled via `config.yaml`:

* **Model:** `gpt2`, `gpt2-medium`, `codegen-350M`
* **Decoding:** `temperature=0.8`, `top_p=0.95`
* **Execution:** `timeout=5.0s`, `sandbox=True`

---

## 📝 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{pandey2025bottleneck,
  title={The Bottleneck Effect: Why Model Scaling Fails for Code Generation},
  author={Pandey, Ashish},
  journal={arXiv preprint},
  year={2025}
}

```

---

## 📧 Contact

**Ashish Pandey** Research Lead | Undergrad Researcher

Email: ashishpandey9818@gmail.com

---

## 🔒 License

MIT License. See `LICENSE` for details.

```

--
