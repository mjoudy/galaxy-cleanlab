# 🧼 Cleanlab Issue Handler for Tabular Datasets - Integrated for Galaxy 

This repository provides a tool that detects and optionally removes or corrects label and data quality issues in tabular datasets using the [Cleanlab](https://github.com/cleanlab/cleanlab) package. It is designed to integrate with the [Galaxy Project](https://galaxyproject.org/), allowing users to improve their data quality **before** running any downstream machine learning or statistical analysis.

---

## ✨ Key Features

- Supports both **classification** and **regression** tasks
- Detects:
  - 🏷️ Noisy labels
  - 🧭 Outliers
  - 🧬 Near-duplicate samples
  - 🔁 Non-i.i.d. data points

---

## 🧠 What is Cleanlab?

[Cleanlab](https://cleanlab.io/) is an open-source Python library for:
- **Data-centric AI** and quality control
- Automatically identifying **label issues, outliers, and duplicates**
- Supporting classification and regression, across tabular, image, text, and audio data

---

## 📂 Repository Structure

```plaintext
.
├── tool/                     # Wrapped Galaxy tool (CLI + XML interface)
│   ├── cleanlab_issue_handler.py
│   └── cleanlab_issue_handler.xml
├── cleanlab_evaluators.py            # Core logic for training and evaluation
├── cleanlab_evaluation.ipynb         # Use case and evaluation of cleanning impact on different models
├── data/                    
|   ├── training_log.csv         # Stores evaluation logs
├── README.md
