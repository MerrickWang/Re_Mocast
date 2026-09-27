import json

import h5py
import numpy as np
import pytest
import torch

from mocast.metrics.csi import ScoreAccumulator, csi_hss_scores
from mocast.tools.audit_h5 import audit
from mocast.tools.diagnose_forecast import diagnose, write_report, operator_probes
from mocast.tools.common import build_model_from_config
from mocast.tests.conftest import small_model_cfg


def test_lead_csi_uses_same_threshold_macro_average_as_overall():
    pred = np.array([100, 10, 0, 0]).reshape(1, 1, 1, 2, 2)
    truth = np.array([100, 100, 10, 0]).reshape(1, 1, 1, 2, 2)
    meter = ScoreAccumulator([5, 50], (), per_lead_time=True, max_lead_time=1)
    meter.update(pred, truth)
    result = meter.compute()
    expected = csi_hss_scores(pred, truth, [5, 50], ())
    assert result["per_leadtime"]["1"]["csi"] == pytest.approx(expected["csi"])
    assert result["per_leadtime"]["1"]["per_threshold"]["50"]["misses"] == 1


def test_operator_probes_and_source_sign():
    probes = operator_probes()
    assert probes["positive_dx_moves_right"]
    assert probes["fixed_zero_flow_20step_mse"] < 1e-8
    assert probes["legacy_bf16_zero_flow_20step_mse"] > 1e-4
    assert probes["half_pixel_20step_peak"] < probes["equivalent_single_10pixel_peak"]
    model = build_model_from_config({"model": small_model_cfg()}).eval()
    initial = torch.zeros(1, 1, 32, 32)
    motion = torch.zeros(1, 4, 2, 32, 32)
    source = torch.full((1, 4, 1, 32, 32), 0.1)
    result = model.reconstruct(initial, motion, source)
    assert torch.allclose(result[0, :, 0, 0, 0], torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_h5_audit_exposes_units_and_exact_cross_split_duplicates(tmp_path):
    path = tmp_path / "test.h5"
    data = np.zeros((25, 8, 8), dtype=np.uint8)
    data[:, 2:4, 2:4] = 200
    with h5py.File(path, "w") as f:
        f.attrs["creator"] = "unit test"
        for name, n in (("train", 4), ("test", 1)):
            g = f.create_group(name)
            g["all_len"] = n
            for i in range(n):
                g[str(i)] = data
    cfg = {"dataset": {"name": "shanghai", "h5_path": str(path),
                       "h5_encoding": "pixel_0_255", "h5_val_fraction": 0.25,
                       "resize": [4, 4], "resize_mode": "bilinear",
                       "sanitize": {"fill": 0, "clip": [0, 70]}}}
    report = audit(cfg, samples_per_split=0)
    assert report["root_attributes"]["creator"] == "unit test"
    assert report["split_sizes"] == {"train": 3, "val": 1, "test": 1}
    assert not any(report["key_overlap"].values())
    assert len(report["duplicate_sequences"]) == 4
    assert report["cross_split_nonempty_frames"]
    candidate = report["splits"]["train"]["candidate_encodings"]
    assert candidate["pixel_0_255"]["outside_0_70_fraction"] == 0
    assert candidate["dbz"]["outside_0_70_fraction"] > 0
    json.dumps(report)


def test_forecast_diagnostic_end_to_end(tmp_path):
    cfg = {
        "model": small_model_cfg(),
        "dataset": {"name": "synthetic", "input_len": 5, "target_len": 4,
                    "n_sequences": 1, "frames_per_sequence": 9,
                    "height": 32, "width": 32, "resize": [32, 32],
                    "filter": {"enabled": False}, "split_config": {"mode": "all"},
                    "cache_index": False},
        "train": {"batch_size": 1},
    }
    ckpt = tmp_path / "model.pt"
    torch.save({"model": build_model_from_config(cfg).state_dict()}, ckpt)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = diagnose(cfg, str(ckpt), torch.device("cpu"))
    finally:
        torch.set_num_threads(threads)
    assert result["n_samples"] == 1 and not result["partial"]
    assert len(result["scores"]["persistence"]["per_leadtime"]) == 4
    write_report(result, str(tmp_path / "reports"))
    assert (tmp_path / "reports" / "lead_skill.png").is_file()
    assert (tmp_path / "reports" / "lead_metrics.csv").is_file()
