# NEUROVISION AI
### AI-Based Retinal Image Analysis Framework for Early Screening of Autism Spectrum Disorder

> **This is an AI-assisted research screening prototype and is not a clinical diagnosis. Results should not be used as a substitute for assessment by qualified healthcare professionals.**

## 1. Overview
NEUROVISION AI is a complete, modular, reproducible research prototype:

```
Retinal Fundus Image → Preprocessing → CNN + ViT Feature Extraction → Feature Fusion
  → Random Forest Feature Ranking → XGBoost Classification → ASD / Non-ASD
  → Severity Estimation (where labelled data are available) → Grad-CAM → Automated Report
```
Nothing is hard-coded: every prediction, metric and plot is produced by real training/inference on the dataset you supply.
The repository ships **without trained models and without results**; until you train, the app says
*"The trained model is not available. Please complete model training before using prediction."* and shows
*"MODEL STATUS: Training dataset required."*

> **Note on the project document.** The project specification/report itself was not available when this software was written; it was built from the project brief (architecture, objectives, rules). Abstract, literature-review conclusions and the exact research-gap wording should be inserted from your document into sections 3–5 below and `frontend/src/pages/About.jsx`.

## 2. Terminology
Screening (not diagnosis) · "AI-assisted screening prediction" · "The model predicts the ASD-associated class based on the provided retinal image."

## 3. Problem statement
ASD is normally identified through behavioural assessment. A non-invasive, image-based screening aid could support earlier referral; retinal fundus imaging has been investigated as a possible correlate. (Replace with the wording of your document.)

## 4. Objectives
Retinal fundus analysis · CNN and ViT feature extraction · CNN+ViT fusion · Random Forest feature ranking · XGBoost classification (SVM comparison) · ASD/Non-ASD classification · severity estimation where labelled · Grad-CAM · confidence score · automated report · evaluation with Accuracy, Precision, Recall, F1, ROC-AUC.

## 5. Research gap
Prior work is limited by small datasets, single-architecture models and weak interpretability; this project evaluates a hybrid CNN+ViT+classical-ML pipeline with visual explanation under a leakage-safe protocol. (Confirm against your literature review.)

## 6. Architecture
| Stage | Implementation |
|---|---|
| Preprocessing | `ml/preprocessing/` – validation, RGB conversion, black-border crop, 224×224 resize, quality heuristics |
| CNN | ResNet50 transfer learning (frozen → optional fine-tune), GAP layer `cnn_features` (2048-d); optional EfficientNetB0 |
| ViT | keras-hub `ViTBackbone` (ImageNet preset `vit_base_patch16_224_imagenet`), [CLS] token `vit_features` (768-d) |
| Fusion | `fuse_features()` – `concat` (default), `l2_concat`, `weighted_concat` |
| Ranking | Random Forest importance, fitted on **train split only**; K chosen on validation |
| Classifiers | XGBoost (main), SVM (comparison) |
| Severity | Separate XGBoost, trained only with real severity labels |
| Explainability | Grad-CAM (CNN), SHAP (XGBoost) |
| App | FastAPI backend + React/Vite frontend + ReportLab PDF |

## 7. Dataset structure
**Format A – folders**
```
dataset/ASD/*.jpg            dataset/Non_ASD/*.jpg
```
or pre-split: `dataset/{train,validation,test}/{ASD,Non_ASD}/`. (No patient IDs ⇒ image-level stratified split + warning.)

**Format B – CSV** (recommended; enables patient-level splitting)
```
image_path,patient_id,label,severity,split
image001.jpg,P001,ASD,Mild,
image002.jpg,P002,Non-ASD,,
image003.jpg,P003,ASD,Moderate,
```
`split` is optional. Labels must be ASD / Non-ASD (unrecognised labels are an error – never guessed). Relative `image_path` values are resolved from the CSV's folder. Supported: JPG, JPEG, PNG, TIFF.

## 8. Installation
Python 3.10–3.12, Node 18+ (tested 22). GPU (≥8 GB VRAM, CUDA-enabled TensorFlow) strongly recommended for training; CPU is fine for inference and the tests. Use Google Colab (GPU runtime) for training – see `notebooks/colab_training.ipynb` – and copy `models/`, `features/selected_features.npy` and `results/` back for local use.
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # tensorflow-cpu on machines without a GPU
cd frontend && npm install
```
Pretrained ImageNet weights are downloaded on first training run (internet required).

## 9. Dataset preparation
```bash
python ml/preprocessing/dataset_validation.py            # or: --csv labels.csv
```
Reports total/valid/corrupted/duplicate/missing images, ASD/Non-ASD counts, class balance, patient count, dimensions, severity distribution → `results/dataset_report.json`; writes `results/split_manifest.csv`.
**Leakage prevention:** with `patient_id` the split is by patient (no patient in two splits; a pre-supplied split that leaks is rejected). Without IDs, a stratified split is used and the limitation is reported. The test split is never used for training, feature selection or tuning and is never re-balanced.

## 10–16. Training pipeline
```bash
python ml/cnn/train_resnet.py                          # 10. CNN baseline → models/resnet50_best.keras
python ml/cnn/extract_cnn_features.py                  #     → features/cnn_features.npy
python ml/vit/train_vit.py                             # ViT baseline → models/vit_best.keras
python ml/vit/extract_vit_features.py                  # 11. → features/vit_features.npy
python ml/fusion/feature_fusion.py                     # 12. → features/fused_features.npy
python ml/feature_selection/random_forest_selection.py # 13. → results/feature_importance.csv, features/selected_features.npy
python ml/classification/train_xgboost.py              # 14. → models/xgboost_asd.json (+ pipeline_meta.json)
python ml/classification/train_svm.py                  #     SVM comparison
python ml/severity/train_severity_xgboost.py           # 15. exits gracefully without real severity labels
python ml/evaluation/evaluate_models.py                # 20. test-set evaluation
python ml/explainability/shap_analysis.py              # optional SHAP
python ml/run_pipeline.py                              # everything, in order
```
* **Random Forest / K:** `TOP_K_FEATURES` (default 500) is configurable; with `AUTO_TOPK=1` (default) K is picked among `TOP_K_CANDIDATES` by validation ROC-AUC (`results/topk_sweep.csv`).
* **Severity:** trained only if every class (Mild/Moderate/Severe) has ≥ `MIN_SEVERITY_SAMPLES_PER_CLASS` real ASD training labels. Otherwise: *Severity: Not Available – the dataset does not contain validated severity labels.* Severity is never synthesised or derived from confidence, and is only reported when ASD is predicted.
* **Grad-CAM (16):** computed on the CNN component (original / heatmap / overlay in `results/gradcam/`). It shows regions influencing the CNN; it does **not** prove they are ASD biomarkers, and it does not explain the final XGBoost decision.
* **Configuration:** `ml/config.py`; every value can be overridden with `NEUROVISION_<NAME>` environment variables (e.g. `NEUROVISION_EPOCHS=5`). Seed `RANDOM_STATE=42`. Configuration, dataset stats, architecture, history, selected features and metrics are saved under `models/`, `features/`, `results/`.
* **Caveat:** CNN/ViT features for *training* images come from networks fitted on those images and can be optimistic; therefore all model/K/hyper-parameter selection uses the validation split only and final numbers come from the untouched test split.

## 17–19. Application
```bash
uvicorn backend.main:app --reload      # http://127.0.0.1:8000  (docs: /docs)
cd frontend && npm run dev             # http://localhost:5173  (proxies /api → :8000)
```
| Endpoint | Purpose |
|---|---|
| `GET /` | service info |
| `GET /health` | status, missing model artefacts |
| `POST /predict` | multipart `file` → prediction, confidence, severity, Grad-CAM, PDF URL (415 bad type, 413 too big, 422 corrupt, 503 untrained) |
| `POST /validate-image` | integrity/format/size/quality check only |
| `GET /metrics` | real evaluation results (empty + status message if not evaluated) |
| `GET /model-info` | pipeline metadata, model status |
| `GET /dataset-info` | dataset validation report |

Uploaded files are stored under random names, analysed, then deleted (`NEUROVISION_KEEP_UPLOADS=1` to keep). Reports contain no filenames or patient information. Inference never trains.

## 20. Evaluation
`evaluate_models.py` scores every experiment on the same held-out test split: Accuracy, Precision, Recall, F1, ROC-AUC, Sensitivity, Specificity, bootstrap 95 % CI for AUC. Outputs: `results/metrics.csv`, `model_comparison.csv`, `confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`, `training_history.png`, `predictions_test.csv`, `evaluation_record.json`. Compared models: CNN · ViT · CNN+ViT (full fused, XGBoost) · CNN+ViT+RF+XGBoost · CNN+ViT+RF+SVM.

## 21. Testing
```bash
pytest
```
Includes an end-to-end test that trains **tiny random-weight models on synthetic images in a temporary directory** to verify the wiring (files, shapes, save/load, API, PDF). Those numbers are meaningless and are never shown outside the test.

## 22. Limitations
Small/single-site data, no external or clinical validation, possible patient leakage when IDs are missing, class imbalance handling but no re-balanced test set, Grad-CAM/SHAP are not medical explanations, severity depends on the supplied labels, ViT preprocessing assumes the keras-hub ImageNet convention ([-1, 1]) – confirm for other presets.

## 23. Ethical considerations
Use only ethically approved, de-identified data with consent; minimise stored identifiers; report negative/poor results; do not deploy for clinical decisions; do not recommend treatment from the output.

## 24. Medical disclaimer
This is an AI-assisted research screening prototype and is not a clinical diagnosis. Results should not be used as a substitute for assessment by qualified healthcare professionals.

## 25. Future work
External multi-centre validation, calibration, uncertainty estimation, patient-level cross-validation, fusion learned end-to-end, additional fundus-specific pretrained backbones, prospective clinical study.
