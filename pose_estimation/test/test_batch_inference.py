"""Batch discovery, resume integrity, and single model initialization."""
import csv
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inference


@pytest.fixture
def videos(tmp_path):
    root = tmp_path / "runs"
    for name in ("a", "b"):
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "robot_with_audio.mp4").touch()
        (folder / "lucy_raw.mp4").touch()
        (folder / "prepared.mp4").touch()
        (folder / "manifest.json").write_text(json.dumps({"status": "completed"}))
    return root


def write_prediction(video, output, checkpoint):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "timestamp_seconds", *inference.JOINT_NAMES])
        writer.writerow([0, 0, *([0] * 14)])
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "source_video": str(video.resolve()), "checkpoint": str(checkpoint.resolve()),
        "joint_names": inference.JOINT_NAMES, "angle_units": "degrees", "frame_count": 1,
        "output_mode": "point"}))


def test_only_completed_final_videos_and_output_paths(videos, tmp_path):
    (videos / "b/manifest.json").write_text(json.dumps({"status": "remote_completed"}))
    output = tmp_path / "joints"
    plan = inference.plan_video_directory(videos, output, Path("checkpoint.pt"))
    assert len(plan) == 1
    assert plan[0][0] == videos / "a/robot_with_audio.mp4"
    assert plan[0][1] == output / "a/robot_with_audio_joints.csv"


def test_valid_exports_skipped_and_broken_pairs_rejected(videos):
    video = videos / "a/robot_with_audio.mp4"
    output = video.with_name("robot_with_audio_joints.csv")
    checkpoint = Path("checkpoint.pt")
    write_prediction(video, output, checkpoint)
    plan = inference.plan_video_directory(videos, None, checkpoint)
    assert [skip for _, _, skip in plan] == [True, False]
    output.with_suffix(".metadata.json").unlink()
    with pytest.raises(ValueError, match="--overwrite"):
        inference.plan_video_directory(videos, None, checkpoint)
    assert not inference.plan_video_directory(videos, None, checkpoint, overwrite=True)[0][2]


def test_wrong_checkpoint_and_truncated_csv_not_skipped(videos):
    video = videos / "a/robot_with_audio.mp4"
    output = video.with_name("robot_with_audio_joints.csv")
    write_prediction(video, output, Path("old.pt"))
    with pytest.raises(ValueError, match="metadata"):
        inference.plan_video_directory(videos, None, Path("new.pt"))
    output.write_text(",".join(["frame_index", "timestamp_seconds", *inference.JOINT_NAMES]) + "\n")
    with pytest.raises(ValueError, match="row count"):
        inference.plan_video_directory(videos, None, Path("old.pt"))


def test_dry_run_never_loads_checkpoint_or_model(videos, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["inference", "--video-dir", str(videos), "--dry-run", "--limit", "1"])
    load, model = Mock(), Mock()
    monkeypatch.setattr(inference.torch, "load", load)
    monkeypatch.setattr(inference, "DINORegressor", model)
    inference.main()
    load.assert_not_called()
    model.assert_not_called()


def test_model_loaded_once_and_rerun_skips_all(videos, tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    monkeypatch.setattr(sys, "argv", ["inference", "--video-dir", str(videos),
                                      "--checkpoint", str(checkpoint), "--device", "cpu"])
    load = Mock(return_value={"config": {}, "model_state_dict": {}})
    model = Mock()
    constructor = Mock(return_value=Mock(to=Mock(return_value=model)))
    monkeypatch.setattr(inference.torch, "load", load)
    monkeypatch.setattr(inference, "DINORegressor", constructor)
    def export(model, video, output, transform, device, mode, config, batch_size, checkpoint, overwrite):
        write_prediction(video, output, checkpoint)
    exporter = Mock(side_effect=export)
    monkeypatch.setattr(inference, "predict_video", exporter)
    inference.main()
    assert load.call_count == constructor.call_count == 1
    assert exporter.call_count == 2
    inference.main()
    assert load.call_count == constructor.call_count == 1
    assert exporter.call_count == 2
