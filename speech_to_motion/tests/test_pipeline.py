"""Exercise timing, leakage, caching, selective gradients and both training stages."""
import copy
import csv
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from speech_to_motion.common import JOINT_NAMES, load_config, read_json, write_json
from speech_to_motion.dataset import MotionDataset, collate_windows, window_starts
from speech_to_motion.inference import export_csv, predict_audio
from speech_to_motion.losses import motion_loss
from speech_to_motion.models.motion_decoder import MotionDecoder, interpolate_features
from speech_to_motion.prepare_dataset import prepare, split_groups


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_duration_split_keeps_sources_together():
    records = [{"id": str(i), "source_group": str(i // 2), "duration_seconds": i + 1}
               for i in range(40)]
    splits = split_groups(records, [0.8, 0.1, 0.1], 42)
    assert splits == split_groups(records, [0.8, 0.1, 0.1], 42)
    groups = [{r["source_group"] for r in rows} for rows in splits.values()]
    assert all(not groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2)))
    total = sum(r["duration_seconds"] for r in records)
    assert abs(sum(r["duration_seconds"] for r in splits["train"]) / total - 0.8) < 0.03
    with pytest.raises(ValueError, match="three original"):
        split_groups(records[:2], [0.8, 0.1, 0.1], 42)


def test_interpolation_and_masked_loss():
    f = torch.tensor([[0.0], [10.0]], requires_grad=True)
    result = interpolate_features(f, torch.tensor([0.0, 1.0]), torch.tensor([-1.0, 0.5, 2.0]))
    assert torch.allclose(result[:, 0], torch.tensor([0.0, 5.0, 10.0]))
    result.sum().backward()
    assert f.grad is not None
    target = torch.zeros(2, 3, 14)
    prediction = target.clone()
    prediction[1, 1:] = 10000
    mask = torch.tensor([[True, True, True], [True, False, False]])
    assert motion_loss(prediction, target, mask) == 0
    assert window_starts(0, 1, 4, 2) == [0.0]
    assert window_starts(0, 9, 4, 2) == [0.0, 2.0, 4.0, 5.0]


def test_decoder_padding_does_not_change_valid_outputs():
    torch.manual_seed(1)
    decoder = MotionDecoder(8, width=16, layers=2, heads=2, feedforward=32, dropout=0).eval()
    features = [torch.randn(4, 8), torch.randn(8, 8)]
    times = [torch.arange(4).float() / 30, torch.arange(8).float() / 30]
    mask = torch.tensor([[True] * 4 + [False] * 4, [True] * 8])
    batched = decoder(features, times, times, mask)
    alone = decoder(features[:1], times[:1], times[:1], mask[:1, :4])
    assert torch.allclose(batched[:1, :4], alone, atol=1e-6)


@pytest.fixture
def prepared(tmp_path):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg/ffprobe required for audio integration")
    cfg = load_config("speech_to_motion/configs/baseline.yaml")
    cfg["data"].update(root=str(tmp_path / "pairs"), provenance=str(tmp_path / "source.json"),
                       training_only=False, ratios=[0.8, 0.1, 0.1],
                       prepared_dir=str(tmp_path / "prepared"), ffmpeg=ffmpeg, ffprobe=ffprobe,
                       window_seconds=0.4, train_stride_seconds=0.2, eval_stride_seconds=0.4)
    cfg["encoder"] = {"name": "tiny-test-hubert", "revision": "test-revision"}
    cfg["decoder"].update(width=16, layers=2, heads=2, feedforward=32, dropout=0.1)
    cfg["training"].update(output_dir=str(tmp_path / "outputs"), device="cpu", precision="fp32",
                           workers=0, batch_size=2, accumulation_steps=2, frozen_epochs=1,
                           finetune_epochs=1, unfreeze_last_n=1, gradient_checkpointing=True)
    clips = []
    for i, duration in enumerate([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        folder = Path(cfg["data"]["root"]) / f"clip-{i}"
        folder.mkdir(parents=True)
        audio = np.sin(np.arange(round(duration * 16000)) * 0.08) * 10000
        with wave.open(str(folder / "source_audio.m4a"), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(audio.astype("<i2").tobytes())
        times = np.arange(0, duration, 1 / 30)
        with (folder / "robot_with_audio_joints.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame_index", "timestamp_seconds", *JOINT_NAMES])
            for n, t in enumerate(times):
                writer.writerow([n, t, *[np.sin(t * (j + 1)) + i for j in range(14)]])
        write_json(folder / "manifest.json", {"status": "completed", "timeline": {"prepared_start_seconds": 0}})
        write_json(folder / "robot_with_audio_joints.metadata.json",
                   {"joint_names": JOINT_NAMES, "angle_units": "degrees", "frame_count": len(times)})
        clips.append({"path": f"clips/clip-{i}.mp4", "url": f"source-{i}", "speaker": "test"})
    write_json(cfg["data"]["provenance"], {"clips": clips})
    summary = prepare(cfg)
    assert not summary["rejected"]
    return cfg


def test_dataset_alignment_and_content_guard(prepared):
    dataset = MotionDataset(prepared, "train")
    item = dataset[0]
    record_index, _ = dataset.windows[0]
    record = dataset.records[record_index]
    from speech_to_motion.prepare_dataset import read_motion
    times, angles = read_motion(dataset.root / record["motion"])
    from speech_to_motion.normalization import denormalize_angles
    denormalized = denormalize_angles(item["target"].numpy(), dataset.normalization)
    assert np.allclose(denormalized[:, 0], np.interp(item["start"] + item["times"].numpy(), times, angles[:, 0]), atol=1e-6)
    assert len(item["audio"]) <= 6400
    path = dataset.root / record["motion"]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after preparation"):
        MotionDataset(prepared, "train")


@pytest.fixture
def tiny_hubert(monkeypatch):
    transformers = pytest.importorskip("transformers")
    config = transformers.HubertConfig(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
        conv_dim=[8, 8, 8], conv_stride=[5, 4, 4], conv_kernel=[10, 8, 4],
        num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2,
        hidden_dropout=0, attention_dropout=0, activation_dropout=0, feat_proj_dropout=0,
        layerdrop=0, apply_spec_augment=False)
    config._commit_hash = "test-revision"
    def load(*args, **kwargs):
        with torch.random.fork_rng():
            torch.manual_seed(12)
            return transformers.HubertModel(copy.deepcopy(config))
    monkeypatch.setattr(transformers.HubertModel, "from_pretrained", load)
    return config


def test_selective_unfreezing_gradients(tiny_hubert):
    from speech_to_motion.models.hubert_encoder import HubertEncoder
    encoder = HubertEncoder(hubert_config=tiny_hubert.to_dict())
    encoder.set_trainable(1, checkpointing=True)
    encoder.train()
    assert not encoder.model.encoder.layers[0].training
    assert encoder.model.encoder.layers[1].training
    features = encoder([torch.randn(4000), torch.randn(3200)])
    sum(x.square().mean() for x, _ in features).backward()
    assert all(p.grad is None for p in encoder.model.feature_extractor.parameters())
    assert all(p.grad is None for p in encoder.model.encoder.layers[0].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.model.encoder.layers[1].parameters())
    encoder.set_trainable(0)
    encoder.train()
    assert not encoder.model.training
    assert not any(p.requires_grad for p in encoder.parameters())


def test_cached_training_resume_finetune_and_export(prepared, tiny_hubert, tmp_path):
    from speech_to_motion.extract_features import extract, read_cache_index
    from speech_to_motion.train import checkpoint_model, load_checkpoint, train
    from speech_to_motion.evaluate import evaluate
    extract(prepared, "cpu", batch_size=3)
    index = read_cache_index(prepared)
    online = MotionDataset(prepared, "train")
    cached = MotionDataset(prepared, "train", index["cache_key"])
    from speech_to_motion.models.speech_to_motion import SpeechToMotion
    model = SpeechToMotion(prepared).eval()
    with torch.no_grad():
        expected = model(collate_windows([online[0]]))
        actual = model(collate_windows([cached[0]]))
    assert torch.allclose(actual, expected, atol=0.002)
    frozen_path = train(prepared, "frozen", "frozen")
    frozen = load_checkpoint(frozen_path)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(frozen_path.parent.parent / "tensorboard" / "epochs")).Reload()
    assert events.Scalars("loss/validation")[0].step == 1
    assert events.Scalars("mae_degrees/val")[0].value >= 0
    assert 0 <= events.Scalars("accuracy_percent/val/within_5_deg")[0].value <= 100
    assert events.Images("accuracy_matrix/val")
    assert events.Scalars("per_joint_rmse_degrees/val/FaceYaw")
    assert events.Scalars("per_joint_rmse_percent_of_range/val/FaceYaw")
    assert events.Images("rmse_by_joint/val")
    assert events.Scalars("loss/train")[0].value == pytest.approx(
        __import__('json').loads((frozen_path.parent.parent / 'metrics.jsonl').read_text().splitlines()[0])['train_loss'])
    prepared["training"]["frozen_epochs"] = 2
    train(prepared, "frozen", "frozen", resume=str(frozen_path.parent / "last.pt"))
    assert load_checkpoint(frozen_path.parent / "last.pt")["epoch"] == 1
    uninterrupted = train(prepared, "frozen", "uninterrupted")
    resumed = load_checkpoint(frozen_path.parent / "last.pt")
    direct = load_checkpoint(uninterrupted.parent / "last.pt")
    assert all(torch.equal(v, direct["model"][k]) for k, v in resumed["model"].items())
    events.Reload()
    assert [event.step for event in events.Scalars("loss/train")] == [1, 2]
    finetune_path = train(prepared, "finetune", "finetune", init_checkpoint=str(frozen_path))
    finetuned = load_checkpoint(finetune_path)
    before = load_checkpoint(frozen_path)
    assert all(torch.equal(v, finetuned["model"][k]) for k, v in before["model"].items()
               if k.startswith("encoder.model.feature_extractor.") or k.startswith("encoder.model.encoder.layers.0."))
    assert any(not torch.equal(v, finetuned["model"][k]) for k, v in before["model"].items()
               if k.startswith("encoder.model.encoder.layers.1."))
    model = checkpoint_model(finetuned).eval()
    times, angles = predict_audio(model, np.sin(np.arange(18000) * 0.1).astype(np.float32), prepared,
                                  finetuned["normalization"], torch.device("cpu"))
    assert len(times) == len(angles) and angles.shape[1] == 14
    assert np.all(np.diff(times) > 0) and np.isfinite(angles).all()
    export_csv(tmp_path / "motion.csv", times, angles, {})
    assert (tmp_path / "motion.csv").read_text().splitlines()[0].split(",") == ["frame_index", "timestamp_seconds", *JOINT_NAMES]
    with pytest.raises(FileExistsError):
        export_csv(tmp_path / "motion.csv", times, angles, {})
    metrics = evaluate(finetuned, prepared, "val", torch.device("cpu"))
    assert np.isfinite(metrics["mae_degrees"])
    assert metrics["frames"] > 0
    assert np.asarray(metrics["accuracy_matrix_percent"]).shape == (14, 4)
    # A partially trainable model must never silently accept cached inputs.
    model.encoder.set_trainable(1)
    with pytest.raises(ValueError, match="cannot use cached"):
        model(collate_windows([cached[0]]))


def test_inference_fractional_origin_has_no_gaps(prepared):
    class ConstantMotion(torch.nn.Module):
        def forward(self, batch):
            return torch.zeros(*batch["mask"].shape, 14)
    normalization = {"mean": [2.0] * 14, "std": [1.0] * 14}
    times, angles = predict_audio(ConstantMotion(), np.zeros(18000, dtype=np.float32), prepared,
                                  normalization, torch.device("cpu"), start=0.00004, end=0.97001)
    assert len(times) > 0 and np.allclose(angles, 2.0)


def test_tensorboard_backfills_existing_history_once(tmp_path):
    import json
    from speech_to_motion.tensorboard_logging import import_history
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run = tmp_path / "existing"
    run.mkdir()
    (run / "metrics.jsonl").write_text(json.dumps(
        {"epoch": 0, "train_loss": 0.7, "val_loss": 0.8, "best_val": 0.8}) + '\n')
    assert import_history(tmp_path) == ["existing"]
    assert import_history(tmp_path) == []
    events = EventAccumulator(str(run / "tensorboard" / "epochs")).Reload()
    assert events.Scalars("loss/validation")[0].value == pytest.approx(0.8)


def test_degree_metrics_denormalize_and_ignore_padding():
    from speech_to_motion.metrics import AngleMetrics
    accumulator = AngleMetrics({"mean": [0] * 14, "std": [2] * 14})
    predicted = torch.tensor([0.0, 1.5, 999.0])[None, :, None].expand(1, 3, 14)
    accumulator.update(predicted, torch.zeros_like(predicted), torch.tensor([[True, True, False]]))
    report = accumulator.compute()
    assert report["mae_degrees"] == pytest.approx(1.5)
    assert report["rmse_degrees"] == pytest.approx(np.sqrt(4.5))
    assert report["accuracy_matrix_percent"] == [[50.0, 50.0, 100.0, 100.0]] * 14
    assert report["per_joint_rmse_degrees"]["FaceYaw"] == pytest.approx(np.sqrt(4.5))
    assert report["per_joint_rmse_percent_of_range"]["FaceYaw"] == pytest.approx(np.sqrt(4.5) / 20 * 100)


def test_physical_normalization_roundtrip_and_legacy_support():
    from speech_to_motion.common import JOINT_LIMITS
    from speech_to_motion.normalization import joint_limit_normalization, normalize_angles, denormalize_angles
    limits = np.asarray(JOINT_LIMITS)
    norm = joint_limit_normalization(limits.mean(1).tolist())
    physical = np.stack([limits[:, 0], limits.mean(1), limits[:, 1]])
    assert np.allclose(normalize_angles(physical, norm), np.array([-1, 0, 1])[:, None])
    assert np.allclose(denormalize_angles(normalize_angles(physical, norm), norm), physical)
    assert np.all(normalize_angles(limits[:, 1] + 1, norm) > 1)  # Matches pose_estimation: no clipping.
    legacy = {"mean": [3] * 14, "std": [2] * 14}
    assert np.allclose(denormalize_angles(np.ones(14), legacy), 5)


def test_migration_keeps_splits_and_reuses_features(prepared, tmp_path):
    from speech_to_motion.common import digest, read_records
    from speech_to_motion.migrate_normalization import migrate
    from speech_to_motion.normalization import training_mean
    source = Path(prepared["data"]["prepared_dir"])
    legacy = {"mean": training_mean(read_json(source / "normalization.json")).tolist(), "std": [2] * 14}
    write_json(source / "normalization.json", legacy)
    splits = {name: read_records(source / "splits" / f"{name}.jsonl") for name in ("train", "val", "test")}
    summary = read_json(source / "summary.json")
    summary["dataset_id"] = digest({"splits": splits, "normalization": legacy})
    write_json(source / "summary.json", summary)
    feature = source / "hubert_features" / "test" / "example.pt"
    feature.parent.mkdir(parents=True)
    feature.write_bytes(b"immutable features")
    write_json(feature.parent / "index.json", {"dataset_id": summary["dataset_id"]})
    destination = tmp_path / "minmax"
    report = migrate(source, destination)
    assert report["dataset_id"] != summary["dataset_id"]
    assert (destination / "splits" / "train.jsonl").read_bytes() == (source / "splits" / "train.jsonl").read_bytes()
    assert read_json(feature.parent / "index.json")["dataset_id"] == summary["dataset_id"]
    assert read_json(destination / "hubert_features/test/index.json")["dataset_id"] == report["dataset_id"]
    assert (destination / "hubert_features/test/example.pt").read_bytes() == feature.read_bytes()
    cfg = copy.deepcopy(prepared)
    cfg["data"]["prepared_dir"] = str(destination)
    assert MotionDataset(cfg, "train").normalization["method"] == "joint_limits"
    with pytest.raises(FileExistsError):
        migrate(source, destination)
    with pytest.raises(ValueError, match="normalization differs"):
        MotionDataset(prepared, "train")


def test_training_only_cache_both_stages_and_no_validation(prepared, tiny_hubert, tmp_path, monkeypatch):
    import importlib
    import json
    from speech_to_motion.extract_features import extract, read_cache_index
    from speech_to_motion.migrate_normalization import migrate
    from speech_to_motion.train import train, load_checkpoint
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    extract(prepared, "cpu", batch_size=3)
    source = Path(prepared["data"]["prepared_dir"])
    destination = tmp_path / "all-clips"
    migrate(source, destination, training_only=True)
    assert read_json(source / "normalization.json") == read_json(destination / "normalization.json")
    cfg = copy.deepcopy(prepared)
    cfg["data"].update(training_only=True, ratios=[1, 0, 0], prepared_dir=str(destination))
    cfg["training"].update(frozen_epochs=2, patience=0)
    assert len(MotionDataset(cfg, "train").records) == 6
    with pytest.raises(ValueError, match="no validation"):
        MotionDataset(cfg, "val")
    with pytest.raises(ValueError, match="Window coverage changed"):
        read_cache_index(cfg)
    extract(cfg, "cpu", batch_size=3)
    assert set(read_cache_index(cfg)["windows"]) == {"train"}
    def forbidden(*args, **kwargs):
        raise AssertionError("Training-only mode must never evaluate validation data")
    monkeypatch.setattr(importlib.import_module('speech_to_motion.train'), 'validate', forbidden)
    frozen = train(cfg, "frozen", "all-frozen")
    last = load_checkpoint(frozen.parent / "last.pt")
    assert last["epoch"] == 1  # No early stop, even with patience=0.
    assert last["selection_metric"] == "train_loss" and "best_val" not in last
    records = [json.loads(line) for line in (frozen.parent.parent / "metrics.jsonl").read_text().splitlines()]
    assert all('val_loss' not in row and 'val_metrics' not in row for row in records)
    tags = EventAccumulator(str(frozen.parent.parent / 'tensorboard/epochs')).Reload().Tags()
    assert all('val' not in tag and 'test' not in tag for tag in tags['scalars'] + tags['images'])
    assert 'mae_degrees/train' in tags['scalars']
    cfg["training"]["frozen_epochs"] = 3
    train(cfg, "frozen", "all-frozen", resume=str(frozen.parent / "last.pt"))
    assert load_checkpoint(frozen.parent / "last.pt")["epoch"] == 2
    finetuned = train(cfg, "finetune", "all-finetune", init_checkpoint=str(frozen.parent / 'last.pt'))
    assert load_checkpoint(finetuned)["selection_metric"] == "train_loss"


def test_prepare_without_splits(prepared, tmp_path):
    cfg = copy.deepcopy(prepared)
    cfg['data'].update(training_only=True, ratios=[1, 0, 0], prepared_dir=str(tmp_path / 'all-prepared'))
    summary = prepare(cfg)
    assert summary['training_only']
    assert summary['splits']['train']['clips'] == 6
    assert summary['splits']['val']['clips'] == summary['splits']['test']['clips'] == 0
