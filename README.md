# Image Classification with Handcrafted Features

An experimental image-classification pipeline covering exploratory analysis, augmentation, handcrafted feature extraction, model selection, evaluation, and submission generation.

## Overview

The final pipeline combines classical computer-vision descriptors with dimensionality reduction and an RBF-kernel support vector machine. The notebooks also preserve comparisons among multiple feature families and classifiers.

Key components include:

- dataset inspection and class-distribution analysis;
- resizing, corner cropping, and image augmentation;
- HOG, EOG, LBP, ORB, SIFT, RGB/HSV histogram, and statistical features;
- standardization and PCA;
- SVM and alternative classifier experiments;
- confusion matrices, classification reports, and CSV submission generation.

## Repository structure

| Path | Description |
| --- | --- |
| `Submission/Final_Model.ipynb` | End-to-end final pipeline and prediction workflow |
| `Submission/Final_Report.pdf` | Final project report |
| `Augmentation & Extraction/` | Data analysis, augmentation, and feature notebooks |
| `fineTuning/` | Classifier, PCA, HOG, and SVM ablations |
| `train_extraction_temp/` | Focused descriptor experiments |
| `result/` | Saved experiment results |
| `train_svm.ipynb` | Standalone PCA + RBF-SVM training experiment |

## Final pipeline

1. Load and inspect the image dataset.
2. Split labeled data into training and validation subsets.
3. Apply augmentation and resizing.
4. Extract handcrafted visual descriptors.
5. Standardize the feature vectors.
6. Retain principal components explaining 75% of the variance.
7. Train a tuned RBF SVM.
8. Evaluate with accuracy, a classification report, and a confusion matrix.
9. Predict the test set and export a submission CSV.

## Environment

The notebooks use Python 3 and the following main packages:

```bash
pip install jupyter numpy pandas pillow matplotlib opencv-python \
  scikit-image scikit-learn scikit-learn-intelex tqdm
```

`scikit-learn-intelex` is optional but is used to accelerate scikit-learn.

## Running the project

```bash
git clone https://github.com/LittleLisa18/Image_classification.git
cd Image_classification
jupyter notebook
```

Place the dataset at the paths expected by the notebook, or update the data-path variables near its beginning. Then run `Submission/Final_Model.ipynb` from top to bottom.

The raw competition dataset is not included. Generated feature arrays can be memory intensive; run extraction incrementally on machines with limited RAM.

## Results

Saved validation reports and tuning logs are available in `classification_report.txt`, `fineTuning/`, and `result/`. The repository retains intermediate experiments to make the final design choices reproducible.

## Project context

This repository was created as an academic image-classification project and is retained as an experimental record of classical computer-vision modeling.
