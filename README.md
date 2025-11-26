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
├── data/                 # Datasets and results
├── models/               # Model checkpoints and configs
├── src/                  # Source code
│   ├── data/            # Data loading
│   ├── models/          # Model wrappers
│   ├── evaluation/      # Execution engine
│   └── analysis/        # Failure analysis
├── scripts/              # Executable scripts
├── notebooks/            # Jupyter notebooks
├── outputs/              # Figures, tables, logs
└── tests/                # Unit tests
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <your-repo-url>
cd paper11_code_execution_failures

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### 2. Download Data

```bash
# Download HumanEval
python scripts/download_data.py --dataset humaneval

# Download MBPP (optional)
python scripts/download_data.py --dataset mbpp
```

### 3. Run Feasibility Check (Week 1-2)

```bash
# Generate samples for 50 problems
python scripts/generate_samples.py --config config.yaml --num_problems 50

# Evaluate and categorize failures
python scripts/run_evaluation.py --config config.yaml

# Analyze failure patterns
python scripts/analyze_failures.py --config config.yaml --output outputs/feasibility_report.json
```

---

## 📊 Pipeline Overview

### Phase 1-2: Feasibility Check (Weeks 1-2)
- Generate 100 samples × 50 problems = 5,000 total samples
- Execute and categorize failures
- **Decision Point**: Proceed if >70% failures are categorizable

### Phase 3-4: Full Analysis (Weeks 3-5)
- Generate 100 samples × 164 problems = 16,400 total samples
- Build failure taxonomy
- Extract activation patterns

### Phase 5-6: Method Development (Weeks 6-8)
- Implement execution-aware decoding
- Compare to baselines
- Write paper

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
@article{yourname2026codeexecution,
  title={Why Code Generation Actually Fails: Execution-Aware Analysis},
  author={Your Name},
  journal={NeurIPS},
  year={2026}
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

## 🗓️ Timeline

- **Week 1-2**: Feasibility check ✅
- **Week 3-5**: Full analysis
- **Week 6-8**: Method + paper writing
- **Target**: NeurIPS 2026 submission (June deadline)
