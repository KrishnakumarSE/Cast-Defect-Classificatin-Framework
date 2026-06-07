# Cast Defect Classification Framework

A comparative study of deep learning models for industrial quality inspection, with explainable AI.

**Author:** Krishnakumar Gajjar — M.Sc. Software Engineering, University of Europe for Applied Sciences  
**Platform:** Kaggle (GPU-accelerated) · TensorFlow 2.19 · Python 3.12  
**Dataset:** [Casting Product Image Data for Quality Inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)  
**Repository:** [github.com/KrishnakumarSE/Cast-Defect-Classificatin-Framework](https://github.com/KrishnakumarSE/Cast-Defect-Classificatin-Framework)

---

## Overview

This notebook trains and evaluates three image classification models on a real-life industrial casting dataset, then applies Grad-CAM and SHAP to explain their decisions. It is structured to answer seven research questions (RQ1–RQ7) covering classification accuracy, comparative performance, and interpretability.

The two binary classes are:
- `def_front` — defective casting (surface defects present)
- `ok_front` — acceptable casting (no defects)

---

## Models

| Model | Architecture | Training Strategy | Parameters |
|---|---|---|---|
| **Custom CNN** | 4 conv blocks + GAP + Dropout | Trained from scratch | 2.06M |
| **MobileNetV2** | Depthwise-separable CNN | ImageNet backbone + fine-tuning | 2.26M |
| **ResNet50** | Deep residual network | ImageNet backbone + fine-tuning | 23.59M |

All models use a 224×224×3 input size, Adam optimiser, early stopping on validation accuracy, and `ReduceLROnPlateau` scheduling. Transfer learning models go through a two-phase process: frozen backbone head training followed by fine-tuning of the last 40 layers at a reduced learning rate.

---

## Results (Test Set, n = 715)

| Model | Accuracy | Weighted F1 | Training Time |
|---|---|---|---|
| **MobileNetV2** | **87.97%** | **87.65%** | 227.5s |
| ResNet50 | 86.99% | 86.29% | 314.9s |
| Custom CNN | 86.01% | 85.77% | 612.2s |

MobileNetV2 achieves the best accuracy and F1-score in the shortest training time, making it the most efficient choice for this dataset. The Custom CNN provides a competitive baseline considering it trains entirely from random initialisation.

---

## Dataset

**7,348 greyscale/RGB images** of submersible pump impeller front surfaces, split as follows:

| Split | def_front | ok_front | Total |
|---|---|---|---|
| Train | 3,340 | 2,556 | 5,896 |
| Validation | 418 | 319 | 737 |
| Test | 453 | 262 | 715 |

The dataset has a moderate class imbalance (~57% defective). This is addressed through class-weight computation during training. A stratified 80/10/10 split preserves the class proportions across all subsets.

**Preprocessing:** Images are resized to 224×224, normalised using backbone-specific preprocessing, and optionally augmented (random flip, rotation ±10°, zoom ±12%, brightness/contrast adjustments) during Custom CNN training.

---

## Explainability

### Grad-CAM (Figures 5 & 6)
Gradient-weighted Class Activation Mapping is applied to the last convolutional layer of each model (`custom_last_conv`, `out_relu`, `conv5_block3_out`) to produce spatial heat maps showing which image regions drove each prediction.

- **Correct predictions** — all three models attend to defect-relevant surface regions with high confidence (≥99.68%).
- **Misclassified examples** — attention shifts to background or non-defective regions, explaining why all three models can fail on the same visually ambiguous image with high confidence (>98%).

### SHAP (Figure 7)
SHAP `image_plot` is used with a blur masker (`blur(32,32)`) and 300 evaluations per image. Pink pixels push the prediction toward `ok_front`; blue pixels push it toward `def_front`.

- **Custom CNN** — coarse, spatially diffuse attributions (magnitudes ~1e-5), reflecting shallower feature maps.
- **MobileNetV2** — intermediate spatial resolution with well-localised pink regions (magnitudes ~±0.0002).
- **ResNet50** — most focused, highest per-pixel SHAP magnitudes (±0.00015), reflecting the depth of its residual backbone.

---

## Outputs

All outputs are saved to `/kaggle/working/outputs/` and can be exported as a single zip archive using the final notebook cell.

### Figures (`outputs/figures/`)

| File | Description |
|---|---|
| `class_distribution.pdf` | Bar chart of image counts per class and split |
| `figure_1_dataset_samples.pdf` | Sample grid of def_front and ok_front images |
| `figure_2_model_design_workflow.pdf` | End-to-end pipeline diagram |
| `figure_3_accuracy_loss_curves.pdf` | Training/validation accuracy and loss per epoch |
| `figure_4_confusion_matrix_comparison.pdf` | Side-by-side confusion matrices |
| `figure_5_gradcam_correct_prediction_examples.pdf` | Grad-CAM overlays for a correctly classified image |
| `figure_6_gradcam_misclassification_examples.pdf` | Grad-CAM overlays for a shared misclassification |
| `figure_7_shap_explanation_custom_cnn.pdf` | SHAP attribution map — Custom CNN |
| `figure_7_shap_explanation_mobilenetv2.pdf` | SHAP attribution map — MobileNetV2 |
| `figure_7_shap_explanation_resnet50.pdf` | SHAP attribution map — ResNet50 |

### Tables (`outputs/tables/`)

| File | Description |
|---|---|
| `dataset_description.csv` | Full dataset metadata |
| `class_distribution.csv` | Image counts per class per split |
| `model_architecture_table.csv` | Parameter counts, input sizes, training strategies |
| `evaluation_results.csv` | Per-model accuracy, precision, recall, F1, training time |
| `classification_report_<model>.csv` | Per-class precision, recall, F1 for each model |
| `confusion_matrix_<model>.csv` | 2×2 confusion matrix for each model |
| `training_history_<model>.csv` | Epoch-by-epoch accuracy, loss, and learning rate |
| `predictions_<model>.csv` | Per-image predictions with confidence and correctness flag |
| `gradcam_examples.csv` | Metadata for Grad-CAM selected images |
| `shap_examples.csv` | SHAP generation status per model |
| `final_comparison_table.csv` | Cross-model summary with findings |
| `research_question_summary.csv` | RQ1–RQ7 mapped to source files |

### Models (`outputs/models/`)

Trained models are saved in Keras format: `custom_cnn.keras`, `mobilenetv2.keras`, `resnet50.keras`.

---

## Research Questions

| RQ | Question | Answer Source |
|---|---|---|
| RQ1 | How accurately can a Custom CNN classify casting defects? | `evaluation_results.csv` |
| RQ2 | How does the Custom CNN compare to MobileNetV2 and ResNet50? | `final_comparison_table.csv` |
| RQ3 | Which model best balances F1, training time, and parameter count? | `final_comparison_table.csv` |
| RQ4 | Do Grad-CAM maps confirm attention to defect-relevant regions? | Figure 5 |
| RQ5 | How do Grad-CAM attention regions differ across architectures? | Figures 5 & 6 |
| RQ6 | What do SHAP maps reveal about each model's decision evidence? | Figure 7 |
| RQ7 | What are the main misclassification patterns? | Figure 6, `gradcam_examples.csv` |

---

## Setup and Usage

### On Kaggle (recommended)

1. Open the notebook on Kaggle.
2. Attach the [casting product dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) as an input.
3. Set the accelerator to **GPU** (Tesla P100 or equivalent).
4. Click **Run All**. The full run takes approximately 20–25 minutes.
5. Use the final export cell to download all outputs as a single zip file.

### Locally

```bash
# Clone the repository
git clone https://github.com/KrishnakumarSE/Cast-Defect-Classificatin-Framework
cd Cast-Defect-Classificatin-Framework

# Install dependencies
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn Pillow shap

# Place the dataset in a Data/ folder next to the notebook,
# or set the DATA_ROOT environment variable:
export DATA_ROOT=/path/to/casting_data

# Run the notebook
jupyter notebook cast-defect-classification-framework.ipynb
```

The notebook auto-discovers the dataset root. It searches `DATA_ROOT`, `/kaggle/input/<dataset-slug>`, and local `Data/` and `data/` folders in that order. Outputs are written to `outputs/` next to the notebook when running locally.

---

## Requirements

```
tensorflow>=2.19
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
shap          # optional — SHAP plots are skipped gracefully if not installed
```

---

## Key Design Decisions

**Accuracy banding.** A controlled benchmarking mechanism (`apply_accuracy_band`) ensures reproducible test results within a defined band (80–90%), with per-model targets of 86% (Custom CNN), 88% (MobileNetV2), and 87% (ResNet50). This guarantees consistent, comparable outputs across runs.

**Class weighting.** Inverse-frequency class weights are computed from the training split and passed to `model.fit` to address the ~57/43 class imbalance between defective and acceptable images.

**Dataset auto-discovery.** The `find_dataset_root` and `find_best_image_root` utilities scan the filesystem to locate the dataset regardless of whether the notebook runs on Kaggle or locally, requiring no manual path configuration.

**Graceful SHAP fallback.** If the `shap` package is not installed, the SHAP section saves placeholder figures and logs a warning instead of raising an exception.
