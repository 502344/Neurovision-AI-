import pandas as pd
import pytest

from ml.preprocessing.dataset_validation import make_splits, normalize_label, validate_dataset
from ml.severity.train_severity_xgboost import severity_data_status
from tests.conftest import make_fundus_like


def _df(n_patients=20, per=3):
    rows = []
    for p in range(n_patients):
        for k in range(per):
            rows.append({"image_path": f"p{p}_{k}.png", "patient_id": f"P{p}", "label": "ASD" if p % 2 else "Non_ASD",
                         "severity": None, "split": None})
    return pd.DataFrame(rows)


def test_patient_level_split_has_no_leakage():
    df, mode = make_splits(_df())
    assert "patient-level" in mode
    per_patient = df.groupby("patient_id")["split"].nunique()
    assert (per_patient == 1).all()
    assert set(df.split) == {"train", "validation", "test"}


def test_split_without_patient_ids_warns_in_mode():
    df = _df(); df["patient_id"] = None
    _, mode = make_splits(df)
    assert "NO patient IDs" in mode


def test_predefined_split_with_leakage_is_rejected():
    df = _df(); df["split"] = ["train", "test"] * (len(df) // 2)
    with pytest.raises(ValueError, match="leakage"):
        make_splits(df)


def test_labels_never_guessed():
    assert normalize_label("Non-ASD") == "Non_ASD" and normalize_label("mystery") is None


def test_validate_dataset_finds_corrupt_duplicate_missing(tmp_path):
    make_fundus_like(tmp_path / "a.png", 1); make_fundus_like(tmp_path / "b.png", 2)
    (tmp_path / "dup.png").write_bytes((tmp_path / "a.png").read_bytes())
    (tmp_path / "bad.png").write_bytes(b"junk")
    rows = [{"image_path": str(tmp_path / f), "patient_id": None, "label": "ASD", "severity": None, "split": None}
            for f in ("a.png", "b.png", "dup.png", "bad.png", "ghost.png")]
    valid, rep = validate_dataset(pd.DataFrame(rows))
    assert rep["valid_images"] == 2 and rep["duplicate_images_removed"] == 1
    assert rep["corrupted_images"] == 1 and rep["missing_files"] == 1
    assert rep["patient_ids_available"] is False and any("patient" in w.lower() for w in rep["warnings"])


def test_severity_unavailable_without_labels():
    meta = pd.DataFrame({"split": ["train"] * 4, "label": ["ASD"] * 4, "severity": [None] * 4})
    ok, why = severity_data_status(meta)
    assert not ok and "no severity labels" in why
