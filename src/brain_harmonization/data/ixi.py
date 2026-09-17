"""Join original IXI demographics to filenames without dropping unmatched records."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PATTERN = re.compile(r"^(IXI\d+)-(Guys|HH|IOP)-(.+)-T1\.nii(?:\.gz)?$", re.I)
SITES = {"GUYS": ("Philips", 1.5), "HH": ("Philips", 3.0), "IOP": ("GE", 1.5)}


def subject_id(value: object) -> str:
    """Normalize integer or IXI-prefixed demographic IDs, rejecting ambiguous values."""
    value = str(value).strip().upper().removeprefix("IXI")
    if not re.fullmatch(r"\d+(?:\.0)?", value):
        raise ValueError(f"Invalid IXI identifier: {value!r}")
    return f"IXI{int(float(value)):03d}"


def read_metadata(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    table.columns = [str(c).strip() for c in table.columns]
    id_col = next((k for k in ("IXI_ID", "subject_id") if k in table), None)
    if id_col is None:
        raise ValueError("Metadata requires IXI_ID or subject_id")
    table["subject_id"] = table[id_col].map(subject_id)
    if table.subject_id.duplicated().any():
        raise ValueError("Duplicate demographic subject IDs; resolve before processing")
    return table


def create_manifest(config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Return the union of imaging and demographic records, plus unrecognized files."""
    root = Path(config["dataset"]["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"MRI directory does not exist: {root}")
    metadata = read_metadata(Path(config["dataset"]["metadata"]))
    images: dict[str, list[tuple]] = {}
    unrecognized = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
            continue
        match = PATTERN.match(path.name)
        if not match:
            unrecognized.append(str(path))
            continue
        sid, site, acquisition = match.groups()
        images.setdefault(subject_id(sid), []).append((path, site.upper(), acquisition))
    if not images:
        raise ValueError("No IXI T1 filenames found. Expected IXI002-Guys-0828-T1.nii.gz")
    lookup = metadata.set_index("subject_id").to_dict("index")
    rows = []
    for sid in sorted(set(images) | set(lookup)):
        files = images.get(sid, [])
        meta = lookup.get(sid, {})
        reasons = []
        status = "PASS"
        if len(files) != 1:
            status = "EXCLUDE"
            reasons.append("missing_image" if not files else "duplicate_images")
        path, site, acquisition = files[0] if files else (None, "UNKNOWN", None)
        age = pd.to_numeric(meta.get("AGE", meta.get("age")), errors="coerce")
        sex = meta.get("SEX_ID (1=m, 2=f)", meta.get("sex", meta.get("SEX_ID")))
        if not meta:
            reasons.append("missing_demographics")
        if not np.isfinite(age) or age <= 0:
            age = np.nan
            reasons.append("missing_or_invalid_age")
        if sex is None or pd.isna(sex):
            reasons.append("missing_sex")
        if reasons and status == "PASS":
            status = "REVIEW"
        scanner, field = SITES.get(site, (None, None))
        rows.append(
            {
                "subject_id": sid,
                "path": str(path) if path else "",
                "site": site,
                "scanner": scanner,
                "field_strength_t": field,
                "scanner_metadata_source": "IXI site description; not per-scan verification",
                "acquisition_id": acquisition,
                "age": age,
                "sex": sex,
                "metadata_json": json.dumps(meta, default=str),
                "all_paths": json.dumps([str(f[0]) for f in files]),
                "status": status,
                "reasons": ";".join(reasons),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest["selected"] = False
    # Round-robin random samples balance available sites without favoring filename order.
    rng = np.random.default_rng(config["seed"])
    eligible = manifest[manifest.status != "EXCLUDE"]
    groups = [list(rng.permutation(g.index)) for _, g in eligible.groupby("site", sort=True)]
    selected = []
    limit = config["subject_limit"] or len(eligible)
    while len(selected) < limit and any(groups):
        for group in groups:
            if group and len(selected) < limit:
                selected.append(group.pop())
    manifest.loc[selected, "selected"] = True
    return manifest, unrecognized
