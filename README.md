# Impact of Visual Degradation on Emotion Recognition

This project analyzes how image quality degradation (noise, blur, desaturation) affects the performance of emotion recognition models. I benchmark a pre-trained SOTA model (**DeepFace**) against a custom fine-tuned CNN (**PyTorch**) on **KDEF** dataset.

## Project Status 🚧

| Task | Description | Link to Notebook |
| :--- | :--- | :--- |
| ✅ **PoC** | Very basic Proof of Concept :) | [Open Notebook 02](notebooks/02_deepface_baseline.ipynb) |
| ✅ **EDA** | Exploratory Data Analysis of KDEF dataset and our degradation model | [Open Notebook 01](notebooks/01_data_exploration.ipynb) [Open as HTML](outputs/reports/01_data_exploration.html) |


## 📂 Project Structure

```text
affective-computing-project/
├── data/                    # Dataset storage
│   ├── raw/                 # Karolinska Directed Emotional Faces
│   ├── grayscale/           # Processed black & white images
│   ├── degraded/            # Images with added noise/blur
│   └── samples/             # Small sample set for quick testing
│
├── notebooks/               # Jupyter notebooks for experiments
├── src/                     # Source code
│   ├── data/                # Scripts for loading and degrading images
│   ├── models/              # DeepFace wrapper and PyTorch model definitions
│   └── evaluation/          # Metrics and confusion matrices
│
├── outputs/                 # Saved models, logs, and generated reports
├── app.py                   # Streamlit demo application
├── pyproject.toml           # Project configuration and dependencies
└── uv.lock                  # Lockfile for reproducible builds

```

## 📥 Data Setup

Before running the project, you must download the datasets and place them in the correct directories.

### 1. KDEF (Karolinska Directed Emotional Faces)

1. Go to [https://kdef.se/download-2/register](https://kdef.se/download-2/register) and register for access.
2. Download the dataset.
3. Extract the contents into a folder named `KDEF` inside `data/raw`.
* *Path should look like:* `data/raw/KDEF/AF01/...`


## 🚀 Installation

This project uses **uv** for fast dependency management.

1. **Install uv** (if not installed):
```bash
pip install uv

```


2. **Sync dependencies:**
This command creates the virtual environment and installs all required packages (PyTorch, DeepFace, OpenCV, Streamlit).
```bash
uv sync

```