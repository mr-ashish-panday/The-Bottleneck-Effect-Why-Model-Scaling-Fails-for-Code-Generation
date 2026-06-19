# Paper 11: Why Code Generation Actually Fails

**Execution-Aware Analysis of Code Generation Model Failures**

---

## 🎯 Project Overview

This repository contains research code for studying **why code generation models fail** through execution-aware analysis. Unlike prior work that treats code generation as a text generation problem, we analyze failures through the lens of **execution semantics**.

### Research Questions
1. How do code generation failures differ from NLP failures?
2. Can we predict failure types from model internals?
3. Can execution-aware decoding reduce crashes without retraining?

---

## 📁 Project Structure

```
paper11_code_execution_failures/
├── data/                 # Datasets and results
├── models/               # Model checkpoints and configs
├── src/                  # Source code
│   ├── data/            # Data loading
│   ├── models/          # Model wrappers
│   ├── evaluation/      # Execution engine
│   └── analysis/        # Failure analysis
├── scripts/              # Executable scripts
├── notebooks/            # Jupyter notebooks
├── outputs/              # Figures, tables, logs
└── tests/                # Unit tests
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
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

### 2. Download Data

python scripts/download_data.py --dataset all

---

🧪 Reproduction

To replicate the results from the paper, follow these steps:

Step 1: Run the Generation Loop (The "16k Experiments")
This script generates solutions and immediately executes them to label "Success" vs "Failure".

To replicate the results from the paper, follow these steps:

Step 1: Run the Generation Loop (The "16k Experiments")
This script generates solutions and immediately executes them to label "Success" vs "Failure".



---

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model selection (`gpt2`, `gpt2-medium`)
- Generation parameters (temperature, top_p)
- Hardware constraints (GPU memory)
- Failure categories

---

## 📈 Experiment Tracking

```bash
# Optional: Use Weights & Biases
pip install wandb
wandb login

# Enable in config.yaml
tracking:
  use_wandb: true
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{pandey2025bottleneck,
  title={The Bottleneck Effect: When Small-Model Scaling Fails for Code Generation},
  author={Pandey, Ashish},
  journal={arXiv preprint},
  year={2025}
}
```

---

## 📧 Contact

- **Author**: Ashish Pandey
- **Email**: ashishpandey9818@gmail.com
- **Institution**: Khwopa College Of Engineering

---

## 🔒 License

MIT License - see LICENSE file for details

---

## ⚠️ Hardware Requirements

- **GPU**: 8-12 GB VRAM (tested on RTX 3060/4070 Ti)
- **RAM**: 16 GB minimum
- **Storage**: 50 GB for data + checkpoints
- **Time**: ~95 GPU hours total (spread over 8 weeks)

---
