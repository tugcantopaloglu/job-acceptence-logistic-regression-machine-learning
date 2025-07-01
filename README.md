# Logistic-Regression Mini-Project 📈  

**Author:** Tuğcan Topaloğlu  

---

## 1. Project Overview
This mini‑project implements **Logistic Regression** from scratch to perform binary classification.  
To boost generalisation and prevent over‑fitting, the model supports:

| Technique | Purpose |
|-----------|---------|
| **L2 regularisation** | Weight shrinkage |
| **Early stopping** | Halt training when validation loss stops improving |
| **Threshold optimisation** | Tune the decision boundary for best F1 / balanced accuracy |

All training, evaluation and visualisation steps are reproducible with a single command.

---

## 2. Repository Layout

```
.
├── data/
│   ├── hw1Data.txt            # raw data
│   ├── train_data.txt         # train split
│   ├── validate_data.txt      # validation split
│   └── test_data.txt          # test split
├── results/                   # created automatically
│   ├── epoch_loss_output.txt  # per-epoch loss
│   ├── weights.txt            # learned weights
│   ├── scores.txt             # accuracy / precision / recall / F1
│   ├── *.png                  # data & loss plots (before/after optimisation)
├── main.py                    # entry point
├── DataHandler.py             # I/O, splitting, plotting helpers
├── LogisticRegression.py      # the core algorithm
└── README.md
```

---

## 3. Algorithms & Techniques

1. **Logistic Regression** – gradient‑descent implementation  
2. **L2 Regularisation** – prevents large weights  
3. **Early Stopping** – stops when validation loss fails to drop for _n_ epochs  
4. **Threshold Search** – picks the cut‑off that maximises F1 (optional)

---

## 4. Getting Started

### 4.1 Requirements

| Package | Tested version |
|---------|----------------|
| NumPy | 2.1.3 |
| Pandas | 2.2.3 |
| Matplotlib | 3.9.2 |
| scikit-learn | 1.5.2 |
| Python | 3.8 + (tested on 3.10 & 3.12) |

Install them in one go:

```bash
pip install -r requirements.txt
```

### 4.2 Run

```bash
python main.py
```

* If the dataset is not yet split, the script performs a **70 / 15 / 15** split automatically and writes the files to `./data`.  
* All output artefacts are placed in `./results`.

---

## 5. Inspecting Results

| File | Description |
|------|-------------|
| `results/scores.txt` | Overall metrics (Accuracy, Precision, Recall, F1). Appended if the file exists. |
| `results/epoch_loss_output.txt` | Training / validation loss per epoch. |
| `results/*Before_*.png` | Loss curves before optimisation. |
| `results/*After_*.png` | Loss curves after early‑stopping + threshold tuning. |
| `results/Data_Plot.png` | Scatter plot of the original data. |

Open the PNG files to visually compare optimisation effects.

---

## 6. Notes & Tips

* Result files are created automatically if absent; otherwise the script appends new runs.  
* Keep datasets under `./data` with the original column order.  
* The project works with newer library versions, but the tested ones are listed for reproducibility.  
* If you encounter an issue, please open an issue — contributions welcome!

Happy experimenting! 🚀
