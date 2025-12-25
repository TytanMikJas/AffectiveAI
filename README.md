# Impact of Visual Degradation on Emotion Recognition

This project analyzes how image quality degradation (noise, blur, desaturation) affects the performance of emotion recognition models. We benchmark a pre-trained SOTA model (**DeepFace**) against a custom fine-tuned CNN (**PyTorch**) on **RAF-DB** and **KDEF** datasets.

## 📂 Project Structure

```text
affective-computing-project/
├── data/                    # Dataset storage
│   ├── raw/                 # Original datasets
│   │   ├── KDEF/            # Karolinska Directed Emotional Faces
│   │   └── RAF/             # Real-world Affective Faces Database
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



### 2. RAF-DB (Real-world Affective Faces Database)

1. Visit [http://whdeng.cn/RAF/model2.html](http://whdeng.cn/RAF/model2.html).
2. Follow the instructions to request access (typically for academic use).
3. Once obtained, extract the dataset into a folder named `RAF` inside `data/raw`.
* *Path should look like:* `data/raw/RAF/basic/...` (or similar structure depending on the archive).



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



## 🛠️ Usage

All commands should be run using `uv run` to ensure they execute within the project environment.

**1. Prepare Data**
Generate grayscale and degraded versions of the datasets found in `data/raw`:

```bash
uv run python src/data/degradation.py

```

**2. Train Custom Model (PyTorch)**
Fine-tune the custom CNN on the training data:

```bash
uv run python src/models/train_custom.py

```

**3. Run Evaluation**
Compare DeepFace and the custom model on the test sets:

```bash
uv run python src/evaluation/evaluate.py

```

**4. Launch Demo App**
Start the interactive Streamlit dashboard to visualize results in real-time:

```bash
uv run streamlit run app.py

```

## 👥 Authors

* Student Team - Wroclaw University of Science and Technology
