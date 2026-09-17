from pathlib import Path

import pandas as pd
import pytest
import yaml

from brain_harmonization.config import load_config
from brain_harmonization.data.ixi import create_manifest, read_metadata, subject_id


def config_file(tmp_path):
    template = Path(__file__).parents[1] / "configs/debug.yaml"
    c = yaml.safe_load(template.read_text())
    c["dataset"].update(root="images", metadata="meta.csv")
    c["output"] = "out"
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(c))
    return p


def test_config_relative_paths_and_validation(tmp_path):
    p = config_file(tmp_path)
    c = load_config(p)
    assert Path(c["dataset"]["root"]) == tmp_path / "images"
    data = yaml.safe_load(p.read_text())
    data["experiment"]["test_fraction"] = 0.9
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="fractions"):
        load_config(p)


@pytest.mark.parametrize("value", [2, "2.0", "IXI002", "002"])
def test_identifier(value):
    assert subject_id(value) == "IXI002"


def test_metadata_union_duplicates_and_selection(tmp_path):
    p = config_file(tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    for name in [
        "IXI001-Guys-001-T1.nii",
        "IXI002-HH-001-T1.nii",
        "IXI002-HH-002-T1.nii",
        "IXI004-IOP-001-T1.nii",
        "other.nii",
    ]:
        (images / name).touch()
    pd.DataFrame({"IXI_ID": [1, 2, 3], "AGE": [30, 40, 50], "SEX_ID (1=m, 2=f)": [1, 2, 1]}).to_csv(
        tmp_path / "meta.csv", index=False
    )
    manifest, unknown = create_manifest(load_config(p))
    m = manifest.set_index("subject_id")
    assert len(m) == 4 and len(unknown) == 1
    assert m.loc["IXI001", "selected"]
    assert m.loc["IXI002", "reasons"] == "duplicate_images"
    assert "missing_image" in m.loc["IXI003", "reasons"]
    assert m.loc["IXI004", "status"] == "REVIEW"
    assert m.loc["IXI001", "field_strength_t"] == 1.5
    pd.testing.assert_frame_equal(manifest, create_manifest(load_config(p))[0])


def test_duplicate_metadata_fails(tmp_path):
    p = tmp_path / "meta.csv"
    pd.DataFrame({"IXI_ID": [1, 1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        read_metadata(p)
