# Diagnostic ICD Project

This repository contains the experimental pipeline for predicting diagnostic ICD codes using the MIMIC-IV dataset. All dependencies are managed using a Conda environment for reproducibility.

---

## 🧰 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Tanman783/Diagnostic_ICD_Proj.git
cd Diagnostic_ICD_Proj
```

### 2. Configure Conda Environment

Choose the command below that matches your operating system:

**For Windows Users (Local):**

```bash
conda env create -f environment.yml
conda activate icd-env

```

**For Linux Users (VS Code/Remote):**

```bash
conda env create -f environment_linux.yml
conda activate ICD

```

---

### 3. 📂 Project Structure

The repository is organized as follows:

* **data/**: Contains processed cohort files (`.csv.gz` and `.pkl`).
* **figures/**: Contains the specific images and plots generated for the final report.
* **src/**: Core project logic, including `preprocessing`, `training`, and `evaluation` modules.
* **src/configs/**: YAML files for hyperparameter grid search.
* **notebooks/**: Standalone experimental scripts converted from notebooks.
* **results/**: Comprehensive experimental outputs including summary metrics and performance plots.
* **feature_importance.ipynb**: Notebook utilized for feature importance analysis.
* **plotting.ipynb**: Notebook utilized for generating and formatting the final visualizations.

---

## ⚠️ Required External Data (Manual Setup)

The following files are excluded from GitHub due to file size limits and must be manually placed in the `data/embeddings/` folder to run the experiments:

**Required Files:**

* ClinGraph_nodes.csv
* ClinVec_icd10cm.csv
* ClinVec_icd10cm_embeddings.csv
* icd-10-cm-2022-0010.csv.gz
* icd-10-cm-2022-0050.csv.gz
* icd-10-cm-2022-0100.csv.gz
* icd-10-cm-2022-1000.csv.gz

**Download Link:** [Click here to download embeddings](https://drive.google.com/file/d/1G0mz7dq_Evg6pIM-CcMAvCEJQsHdd6dM/view?usp=sharing)

---

## 🚀 Running Scripts
All experimental scripts in the notebooks/ folder include logic to automatically detect the project root:

```bash
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent 
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
Crucial: You must execute these scripts from the project root directory (Diagnostic_ICD_Proj) to avoid module import errors. Example:
```

```bash
python notebooks/your_script_name.py
```



