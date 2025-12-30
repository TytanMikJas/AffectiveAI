# Impact of Visual Degradation on Emotion Recognition

This project analyzes how image quality degradation (noise, blur - degradations typical to old photos) affects the performance of emotion recognition models. I benchmark a pre-trained SOTA model (**DeepFace**) against a MobileNetV3 customly trained (**PyTorch**) on **KDEF** dataset.

## Project Status ✅

| Task | Description | Link to Notebook |
| :--- | :--- | :--- |
| ✅ **PoC** | Very basic Proof of Concept :) | [Open Notebook 00](notebooks/00_deepface_baseline.ipynb) |
| ✅ **Data Generation** | Obtain and process KDEF dataset using custom degradation pipeline (adapted from my ongoing research paper on "Old Movie Restoration") | [Open Sample Images](data/samples/original_gray_degraded_sample.png) |
| ✅ **EDA** | Exploratory Data Analysis of KDEF dataset and our degradation model | [Open Notebook 01](notebooks/01_data_exploration.ipynb) [Open as generated HTML](outputs/reports/01_data_exploration.html) |
| ✅ **Model Training** | Train MobileNetV3 | [Open Notebook 02](notebooks/02_training_mobilenet_v3.ipynb) |
| ✅ **Evaluation in Numbers** | Benchmark DeepFace vs Custom Model (transfer learning vs fine-tune) across 3 domains (Raw/Gray/Degraded).  Metrics, Confusion Matrices, Overall analysis & Demographic Bias Check | [Open Notebook 03](notebooks/03_evaluation.ipynb) |
| ✅ **Visual Analysis** | Visualize how models performs on certain images | [Open Notebook 04](notebooks/04_visual_analysis.ipynb) |
| ✅ **Demo App** | Streamlit dashboard for real-time Emotion Recognition with grayscale and degraded images | [Open App](data/samples/streamlit_sample.png) |

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