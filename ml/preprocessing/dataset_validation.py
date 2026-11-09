"""Dataset validation, patient-level splitting and split-manifest creation.

Supports
  Format A: dataset/ASD/*.jpg + dataset/Non_ASD/*.jpg  (or pre-split train/validation/test/<class>/)
  Format B: CSV with columns image_path,patient_id,label,severity

Run:  python ml/preprocessing/dataset_validation.py [--csv path.csv] [--root dir]

Outputs
  results/dataset_report.json   (statistics)
  results/split_manifest.csv    (image_path, patient_id, label, severity, split)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml import config  # noqa: E402
from ml.preprocessing.image_preprocessing import InvalidImageError, validate_image  # noqa: E402

LABEL_ALIASES = {"asd": "ASD", "autism": "ASD", "1": "ASD",
                 "non-asd": "Non_ASD", "non_asd": "Non_ASD", "nonasd": "Non_ASD", "control": "Non_ASD",
                 "td": "Non_ASD", "typical": "Non_ASD", "0": "Non_ASD"}
SEVERITY_ALIASES = {"mild": "Mild", "moderate": "Moderate", "severe": "Severe"}
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "validation", "validation": "validation",
                 "valid": "validation", "test": "test"}


def normalize_label(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return LABEL_ALIASES.get(str(value).strip().lower())


def normalize_severity(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return SEVERITY_ALIASES.get(str(value).strip().lower())


# ------------------------------------------------------------------ loaders
def _list_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in config.ALLOWED_EXTENSIONS)


def load_manifest_from_folders(root: Path) -> pd.DataFrame:
    """Format A. Uses pre-made train/validation/test folders if present, else flat class folders."""
    rows = []
    presplit = [s for s in ("train", "validation", "test") if (root / s).is_dir()]
    if presplit:
        for split in presplit:
            for cls in ("ASD", "Non_ASD"):
                for p in _list_images(root / split / cls) if (root / split / cls).is_dir() else []:
                    rows.append({"image_path": str(p), "patient_id": None, "label": cls,
                                 "severity": None, "split": split})
    if not rows:
        for cls in ("ASD", "Non_ASD"):
            if (root / cls).is_dir():
                for p in _list_images(root / cls):
                    rows.append({"image_path": str(p), "patient_id": None, "label": cls,
                                 "severity": None, "split": None})
    return pd.DataFrame(rows, columns=["image_path", "patient_id", "label", "severity", "split"])


def load_manifest_from_csv(csv_path: Path, base_dir: Path | None = None) -> pd.DataFrame:
    """Format B. Required: image_path,label. Optional: patient_id,severity,split."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"image_path", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {sorted(missing)}")
    base_dir = base_dir or csv_path.parent
    df["image_path"] = df["image_path"].astype(str).map(
        lambda s: s if Path(s).is_absolute() else str((base_dir / s).resolve()))
    for col in ("patient_id", "severity", "split"):
        if col not in df.columns:
            df[col] = None
    df["label_raw"] = df["label"]
    df["label"] = df["label"].map(normalize_label)
    df["severity"] = df["severity"].map(normalize_severity)
    df["split"] = df["split"].map(lambda s: SPLIT_ALIASES.get(str(s).strip().lower()) if pd.notna(s) else None)
    df["patient_id"] = df["patient_id"].map(lambda s: str(s).strip() if pd.notna(s) and str(s).strip() else None)
    unknown = df[df["label"].isna()]
    if len(unknown):
        raise ValueError(f"{len(unknown)} row(s) have unrecognised labels, e.g. {unknown['label_raw'].head(3).tolist()}. "
                         "Use ASD / Non-ASD. Labels are never guessed.")
    return df[["image_path", "patient_id", "label", "severity", "split"]]


# ---------------------------------------------------------------- validation
def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Check every file. Returns (valid-only dataframe, statistics report)."""
    total = len(df)
    missing, corrupted, valid_rows, dims = [], [], [], []
    for row in df.itertuples(index=False):
        p = Path(row.image_path)
        if not p.exists():
            missing.append(str(p))
            continue
        try:
            info = validate_image(p, max_mb=1e9)
            dims.append((info["width"], info["height"]))
            valid_rows.append(row._asdict())
        except InvalidImageError as exc:
            corrupted.append({"path": str(p), "reason": str(exc)})
    valid = pd.DataFrame(valid_rows, columns=df.columns)

    # exact duplicate detection by content hash
    duplicates = []
    if len(valid):
        valid["_md5"] = valid["image_path"].map(_md5)
        dup_mask = valid.duplicated("_md5", keep="first")
        duplicates = valid.loc[dup_mask, "image_path"].tolist()
        # duplicates whose labels disagree are an integrity problem worth flagging
        conflicting = valid.groupby("_md5")["label"].nunique()
        conflicting = int((conflicting > 1).sum())
        valid = valid.loc[~dup_mask].drop(columns="_md5").reset_index(drop=True)
    else:
        conflicting = 0

    counts = valid["label"].value_counts().to_dict() if len(valid) else {}
    asd, non = int(counts.get("ASD", 0)), int(counts.get("Non_ASD", 0))
    has_patients = bool(len(valid) and valid["patient_id"].notna().all())
    report = {
        "total_images_listed": total,
        "valid_images": int(len(valid)),
        "corrupted_images": len(corrupted),
        "corrupted_details": corrupted[:50],
        "duplicate_images_removed": len(duplicates),
        "duplicate_details": duplicates[:50],
        "duplicates_with_conflicting_labels": conflicting,
        "missing_files": len(missing),
        "missing_details": missing[:50],
        "ASD_count": asd,
        "Non_ASD_count": non,
        "class_balance": {"ASD": round(asd / max(asd + non, 1), 4), "Non_ASD": round(non / max(asd + non, 1), 4)},
        "patient_ids_available": has_patients,
        "patient_count": int(valid["patient_id"].nunique()) if has_patients else None,
        "image_dimensions": {
            "min": [int(min(d[0] for d in dims)), int(min(d[1] for d in dims))] if dims else None,
            "max": [int(max(d[0] for d in dims)), int(max(d[1] for d in dims))] if dims else None,
            "median": [int(np.median([d[0] for d in dims])), int(np.median([d[1] for d in dims]))] if dims else None,
        },
        "severity_distribution": (valid.loc[valid["label"] == "ASD", "severity"].fillna("Unlabelled")
                                  .value_counts().to_dict() if len(valid) else {}),
        "warnings": [],
    }
    w = report["warnings"]
    if not has_patients:
        w.append("No patient_id column/values: splitting is image-level and stratified. Patient-level leakage "
                 "cannot be completely excluded. Metrics may be optimistic.")
    if min(asd, non) == 0 and len(valid):
        w.append("Only one class present: supervised ASD/Non-ASD classification is impossible.")
    elif len(valid) and min(asd, non) / max(asd, non) < 0.5:
        w.append("Class imbalance detected. The test set is NOT re-balanced; interpret accuracy carefully and "
                 "prefer ROC-AUC, recall/specificity and F1.")
    if len(valid) < 200:
        w.append(f"Small dataset (n={len(valid)}). Estimates will have wide uncertainty; no external validation.")
    if conflicting:
        w.append(f"{conflicting} duplicate image group(s) carry conflicting labels -- inspect the source data.")
    if not report["severity_distribution"] or set(report["severity_distribution"]) == {"Unlabelled"}:
        w.append("No validated severity labels: the severity model will not be trained.")
    return valid, report


# ----------------------------------------------------------------- splitting
def make_splits(df: pd.DataFrame, seed: int | None = None) -> tuple[pd.DataFrame, str]:
    """Assign train/validation/test. Patient-level when patient_id exists, else stratified.

    If every row already has a split, it is respected -- but patient overlap across splits is checked.
    """
    from sklearn.model_selection import train_test_split

    seed = config.RANDOM_STATE if seed is None else seed
    df = df.copy()
    if df["split"].notna().all():
        mode = "pre-defined split (respected)"
        if df["patient_id"].notna().all():
            g = df.groupby("patient_id")["split"].nunique()
            leaked = g[g > 1].index.tolist()
            if leaked:
                raise ValueError(f"Patient leakage in provided split: {len(leaked)} patient(s) appear in several "
                                 f"splits (e.g. {leaked[:5]}). Fix the split before training.")
        return df, mode

    df["split"] = None
    tf_, vf_, sf_ = config.TRAIN_FRACTION, config.VAL_FRACTION, config.TEST_FRACTION
    assert abs(tf_ + vf_ + sf_ - 1.0) < 1e-6, "Split fractions must sum to 1"
    if df["patient_id"].notna().all():
        # one row per patient, label = patient's label (patients with mixed labels are an error)
        pl = df.groupby("patient_id")["label"].nunique()
        if (pl > 1).any():
            raise ValueError(f"Patients with more than one label: {pl[pl > 1].index.tolist()[:5]}")
        pat = df.groupby("patient_id")["label"].first().reset_index()
        strat = pat["label"] if pat["label"].value_counts().min() >= 3 else None
        tr, rest = train_test_split(pat, test_size=vf_ + sf_, stratify=strat, random_state=seed)
        strat2 = rest["label"] if strat is not None and rest["label"].value_counts().min() >= 2 else None
        va, te = train_test_split(rest, test_size=sf_ / (vf_ + sf_), stratify=strat2, random_state=seed)
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            df.loc[df["patient_id"].isin(part["patient_id"]), "split"] = name
        mode = "patient-level stratified split (no patient in more than one split)"
    else:
        strat = df["label"] if df["label"].value_counts().min() >= 3 else None
        tr, rest = train_test_split(df, test_size=vf_ + sf_, stratify=strat, random_state=seed)
        strat2 = rest["label"] if strat is not None and rest["label"].value_counts().min() >= 2 else None
        va, te = train_test_split(rest, test_size=sf_ / (vf_ + sf_), stratify=strat2, random_state=seed)
        df.loc[tr.index, "split"] = "train"
        df.loc[va.index, "split"] = "validation"
        df.loc[te.index, "split"] = "test"
        mode = "image-level stratified split (NO patient IDs: leakage cannot be excluded)"
    return df, mode


def build_manifest(csv: str | None = None, root: str | None = None, save: bool = True):
    """Full pipeline: load -> validate -> split -> save manifest + report. Returns (manifest, report)."""
    config.ensure_dirs()
    csv = csv or config.DATASET_CSV
    if csv:
        base = Path(root or config.DATASET_ROOT) if (root or config.DATASET_ROOT) else None
        raw = load_manifest_from_csv(Path(csv), base)
        source = f"csv:{csv}"
    else:
        raw = load_manifest_from_folders(Path(root or config.DATASET_DIR))
        source = f"folders:{root or config.DATASET_DIR}"
    if raw.empty:
        raise FileNotFoundError("No images found. Provide dataset/ASD + dataset/Non_ASD (or train/validation/test "
                                "sub-folders), or a CSV via --csv. See README 'Dataset preparation'.")
    valid, report = validate_dataset(raw)
    if valid.empty:
        raise ValueError("No valid images after validation.")
    manifest, mode = make_splits(valid)
    report["source"] = source
    report["split_strategy"] = mode
    report["split_counts"] = {s: {c: int(((manifest.split == s) & (manifest.label == c)).sum())
                                  for c in ("ASD", "Non_ASD")} for s in ("train", "validation", "test")}
    for s, c in report["split_counts"].items():
        if c["ASD"] == 0 or c["Non_ASD"] == 0:
            report["warnings"].append(f"Split '{s}' lacks one class ({c}); metrics for it are undefined.")
    manifest["label_idx"] = manifest["label"].map({"Non_ASD": 0, "ASD": 1})
    if save:
        manifest.to_csv(config.SPLIT_MANIFEST, index=False)
        (config.RESULTS_DIR / "dataset_report.json").write_text(json.dumps(report, indent=2, default=str))
    return manifest, report


def load_manifest() -> pd.DataFrame:
    """Read the saved manifest (created by this script)."""
    if not config.SPLIT_MANIFEST.exists():
        raise FileNotFoundError("results/split_manifest.csv not found. Run: python ml/preprocessing/dataset_validation.py")
    return pd.read_csv(config.SPLIT_MANIFEST)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    try:
        _, report = build_manifest(a.csv, a.root)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[dataset_validation] ERROR: {exc}")
        sys.exit(1)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_details")}, indent=2))
    print(f"\nManifest written to {config.SPLIT_MANIFEST}")


if __name__ == "__main__":
    main()
