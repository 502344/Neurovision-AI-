# Run instructions (quick reference)

```bash
# 0. Install
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
(cd frontend && npm install)

# 1. Put your labelled data in dataset/ (folders) OR prepare a CSV, then validate + split
python ml/preprocessing/dataset_validation.py                   # folders
python ml/preprocessing/dataset_validation.py --csv labels.csv  # CSV (image_path,patient_id,label,severity)

# 2. Train (use a GPU / Google Colab -- see notebooks/colab_training.ipynb)
python ml/cnn/train_resnet.py
python ml/cnn/extract_cnn_features.py
python ml/vit/train_vit.py
python ml/vit/extract_vit_features.py
python ml/fusion/feature_fusion.py
python ml/feature_selection/random_forest_selection.py
python ml/classification/train_xgboost.py
python ml/classification/train_svm.py
python ml/severity/train_severity_xgboost.py      # exits gracefully if no real severity labels
python ml/evaluation/evaluate_models.py
python ml/explainability/shap_analysis.py          # optional
#   or all at once:  python ml/run_pipeline.py [--csv labels.csv]

# 3. Run the application
uvicorn backend.main:app --reload        # terminal 1  -> http://127.0.0.1:8000/docs
cd frontend && npm run dev               # terminal 2  -> http://localhost:5173

# 4. Tests
pytest
```
