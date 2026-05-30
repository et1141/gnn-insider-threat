# Deep Learning Project

**Teams:** 2–3 students
**Duration:** 12 meetings
**Goal:** Solve a real-world problem using Deep Learning and follow modern ML engineering (DevOps / MLOps) practices.


---

# 1. Project Goal

The goal of the project is to:

* Apply Deep Learning to a real-life problem
* Build a “fully” reproducible training/evaluation pipeline
* Follow basic DevOps / MLOps standards
* Conduct systematic experimentation
* Clearly communicate results

This project evaluates both:

* Modeling skills
* Engineering quality


---

# 2. Problem Requirements

The project must:

* Solve a real-world prediction problem
* Use Deep Learning as the main method
* Use a public dataset
* Be reproducible

Allowed domains:

* Computer Vision (classification, segmentation, detection)
* NLP (text classification, QA, summarization)
* Time series forecasting
* Audio classification
* Multimodal learning
* Graph neural networks

  

Datasets must come from:

* Kaggle
* HuggingFace Datasets
* OpenML
* UCI ML Repository
* Papers With Code
* Public government datasets
* AWS Open Data

Large datasets:

* Provide download script
* Do not upload raw data to GitHub


---

# 3. Mandatory Engineering & DevOps Requirements

This is a core part of the course.


---

## 3.1 GitHub Repository

Public repository required.

Must include:

```none
project/
├── src/
├── configs/
├── tests/
├── .github/workflows/
├── pyproject.toml
├── README.md
```


:::warning
Notebook-only repositories are **NOT allowed**.

Code must be modular (not one long notebook).

:::

## 3.2 Environment Management (uv)

You must:

* Use `uv`
* Provide installation instructions
* Ensure the project runs via:

uv sync
uv run python src/train.py


---

## 3.3 CI/CD (GitHub Actions)

Your repository must:

* Run tests automatically
* Run a linter (ruff or flake8)
* Pass CI before final submission


---

## 3.4 Testing (pytest)

Minimum required tests:

* Data loading test
* Model forward pass test
* Loss computation test
* Simple training smoke test


---

## 3.5 Experiment Tracking (Weights & Biases)

You must use W&B to log:

* Hyperparameters
* Metrics
* Training curves
* At least 5 experiments
* Best model artifact

W&B project link must be in README.


---

## 3.6 Reproducibility

You must:

* Set random seeds
* Use fixed data splits
* Save best model
* Use configuration files (YAML or similar)
* Ensure experiments can be repeated


---

# 4. Kedro (Recommended)

Students are strongly encouraged to use **Kedro** to structure their project.

Why Kedro?

* Clear pipeline structure
* Separation of data, models, and configuration
* Reproducible pipelines
* Production-ready project organization

Kedro is not mandatory, but:

* Well-structured Kedro projects may receive better evaluation in the Engineering & DevOps category.
* If not using Kedro, students must ensure similar pipeline clarity and modularity.

# 5. What Must Be Presented

Your project must clearly show:


1. Problem definition
2. Dataset description
3. Data preprocessing
4. Model architecture
5. Training setup
6. Experimental comparison
7. Error analysis
8. Final conclusions and limitations