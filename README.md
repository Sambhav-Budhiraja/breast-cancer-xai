# Breast Cancer Ultrasound Classification with Explainable AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sambhav-Budhiraja/breast-cancer-xai/blob/main/LUSIP_2026.ipynb)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)

Deep learning pipeline that classifies breast ultrasound images as **benign**, **malignant** or **normal**, and uses **Grad-CAM, SHAP and LIME** to show which image regions drive each prediction. Built as part of **LUSIP 2026**, following methods from published research papers.

## Overview

Four models are trained and compared on the same data split:

- **VGG16**: transfer learning, frozen convolutional base, custom classifier head
- **ResNet50**: transfer learning, custom head (2048 → 1024 → 3) with dropout, class-weighted loss, hyperparameter search
- **MobileNetV2**: transfer learning, lightweight baseline
- **Custom CNN**: 4-block CNN trained from scratch (no pretrained weights)

## Dataset

[Breast Ultrasound Images Dataset (BUSI)](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset) from Kaggle: 780 ultrasound images.

| Class | Images |
|---|---|
| Benign | 437 |
| Malignant | 210 |
| Normal | 133 |

Segmentation masks are excluded; only the images are used for classification. The dataset downloads automatically in the notebook via `kagglehub`.

## Method

- **Preprocessing:** resize to 224×224, ImageNet normalization
- **Augmentation (train only):** random horizontal flip, random rotation (±15°)
- **Split:** 70% train / 15% validation / 15% test (fixed seed 42, so the test set is 117 images)
- **Training:** Adam optimizer, cross-entropy loss, pretrained ImageNet weights with frozen backbones
- **ResNet50 tuning:** compared learning rates (1e-5, 1e-4, 1e-3), batch sizes (16, 32, 64) and epochs (5, 10, 15)
- **Evaluation:** accuracy, precision, recall, F1, confusion matrix, ROC-AUC

## Results

Evaluated on the held-out test set (117 images):

| Model | Test Accuracy | Precision | Recall | F1 | Transfer Learning |
|---|---|---|---|---|---|
| **VGG16** | **86.32%** | 0.87 | 0.86 | 0.86 | Yes |
| ResNet50 | 73.50% | 0.74 | 0.73 | 0.73 | Yes |
| MobileNetV2 | 70.09% | 0.70 | 0.70 | 0.70 | Yes |
| Custom CNN | 68.37% | 0.71 | 0.68 | 0.66 | No |

Precision, recall and F1 are weighted averages. ResNet50 also reached a macro ROC-AUC of 0.879.

**VGG16 per-class results:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Benign | 0.84 | 0.95 | 0.89 |
| Malignant | 0.94 | 0.75 | 0.83 |
| Normal | 0.81 | 0.85 | 0.83 |

## Explainability

Grad-CAM, SHAP (GradientExplainer) and LIME are applied to ResNet50, MobileNetV2 and VGG16.

- **Grad-CAM** highlights the regions of the ultrasound the network focuses on, using the last convolutional layer.
- **SHAP** assigns pixel-level contribution scores for the predicted class.
- **LIME** perturbs superpixels to find the regions that most support the prediction.

## How to run

**Option 1: Google Colab (easiest).** Click the "Open in Colab" badge above, set the runtime to GPU (Runtime → Change runtime type), and run all cells.

**Option 2: Locally.**

```bash
git clone https://github.com/Sambhav-Budhiraja/breast-cancer-xai.git
cd breast-cancer-xai
pip install -r requirements.txt
jupyter notebook LUSIP_2026.ipynb
```

The notebook downloads BUSI through `kagglehub`. You may need a free Kaggle account. A GPU is strongly recommended.

## Limitations

- The dataset is small (780 images) and the test set has only 117 images (20 normal), so results have high variance.
- Hyperparameter comparisons for ResNet50 were judged on training metrics, so they are indicative only.
- Models were not trained under identical settings (for example, class weighting was used for ResNet50 only), so the comparison is not a strictly controlled benchmark.
- Malignant recall for the best model is 0.75, which would need to be higher before any clinical relevance.
- This is an educational and research project. It is **not a medical device** and must not be used for diagnosis.

## Tech stack

Python, PyTorch, torchvision, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, OpenCV, Grad-CAM, SHAP, LIME

## Author

**Sambhav Budhiraja**
GitHub: [@Sambhav-Budhiraja](https://github.com/Sambhav-Budhiraja)
