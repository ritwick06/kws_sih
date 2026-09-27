#!/usr/bin/env python3
"""Train the Hey Wakey model from the current raw dataset layout.

This entry point reuses the feature, model, evaluation, and export pipeline
from train_v3.py, but reads only local files under:

    kws_dataset/raw/positive/<label>/<speaker>/*.wav
    kws_dataset/raw/negative/<category>/<speaker>/*.wav

No external dataset download or TTS generation is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import train_v3 as pipeline


RAW_CATEGORIES = {
    "positive": {"label": 1, "hard": False},
    "negative": {"label": 0, "hard": False},
}
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".m4a", ".mp3"}


def collect_raw_recordings(data_root: str | Path) -> list[dict]:
    """Build the manifest from raw category-first speaker directories."""
    raw_root = Path(data_root).expanduser().resolve() / "raw"
    entries = []

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_root}")

    for split_name, split_meta in RAW_CATEGORIES.items():
        split_root = raw_root / split_name
        if not split_root.is_dir():
            continue

        for category_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
            for speaker_root in sorted(path for path in category_root.iterdir() if path.is_dir()):
                for audio_path in sorted(speaker_root.rglob("*")):
                    if not audio_path.is_file() or audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
                        continue

                    speaker = speaker_root.name
                    category = f"{split_name}/{category_root.name}"
                    entries.append({
                        "path": str(audio_path),
                        "label": split_meta["label"],
                        "hard": category_root.name == "hard_negative",
                        "group": speaker,
                        "fine_group": f"{speaker}__{audio_path.stem}",
                        "category": category,
                    })

    if not entries:
        raise RuntimeError(f"No audio files found below {raw_root}")

    return entries


def build_raw_manifest(data_root: str | Path) -> tuple[list[dict], list[str]]:
    manifest = collect_raw_recordings(data_root)
    noise_files = [
        entry["path"]
        for entry in manifest
        if entry["category"].startswith("negative/background")
    ]

    positives = sum(entry["label"] for entry in manifest)
    print(
        f"Found {len(manifest)} raw clips: {positives} positive, "
        f"{len(manifest) - positives} negative, "
        f"{len({entry['group'] for entry in manifest})} speakers."
    )
    return manifest, noise_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Hey Wakey from kws_dataset/raw without downloads."
    )
    parser.add_argument(
        "--data-root",
        default="./kws_dataset",
        help="Dataset root containing raw/.",
    )
    parser.add_argument(
        "--artifacts",
        default="./artifacts_raw",
        help="Output directory for models and reports.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of validation bootstrap resamples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.skip_speech_commands = True
    args.skip_tts = True
    args.tts_per_voice = 0
    pipeline.configure_from_args(args)
    pipeline.CFG.RECORDINGS_ROOT = str(Path(args.data_root).resolve() / "raw")
    pipeline.CFG.USE_SPEECH_COMMANDS = False
    pipeline.CFG.GENERATE_SYNTHETIC_POSITIVES = False

    manifest, noise_files = build_raw_manifest(args.data_root)
    train_idx, val_idx, test_idx = pipeline.create_main_split(manifest)
    datasets = pipeline.build_feature_datasets(
        manifest,
        noise_files,
        train_idx,
        val_idx,
        test_idx,
    )
    X_train, y_train, X_val, y_val, X_test, y_test = datasets

    (
        model,
        X_train_n,
        X_val_n,
        X_test_n,
        train_mean,
        train_std,
        n_params,
    ) = pipeline.train_final_model(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    threshold, sweep, val_probs = pipeline.select_validation_threshold(
        model,
        X_val_n,
        y_val,
    )
    bootstrap_report = pipeline.bootstrap_metrics(
        y_val,
        val_probs,
        threshold,
        n_bootstrap=args.bootstrap_samples,
    )

    artifact_dir = Path(pipeline.CFG.ARTIFACT_DIR)
    (artifact_dir / "validation_bootstrap_report.json").write_text(
        pipeline.json.dumps(bootstrap_report, indent=2),
        encoding="utf-8",
    )
    sensitivity_report = pipeline.threshold_sensitivity_report(sweep, threshold)
    (artifact_dir / "threshold_sensitivity_report.json").write_text(
        pipeline.json.dumps(sensitivity_report, indent=2),
        encoding="utf-8",
    )

    recall, false_activation_rate, precision = pipeline.evaluate_float_model(
        model,
        X_test_n,
        y_test,
        threshold,
        sweep,
        n_params,
    )
    pipeline.save_firmware_contract(train_mean, train_std)
    tflite_path = pipeline.export_int8_tflite(model, X_train_n)
    pipeline.evaluate_quantized_model(
        tflite_path,
        X_test_n,
        y_test,
        threshold,
        recall,
        false_activation_rate,
    )
    pipeline.export_c_array(tflite_path)

    (artifact_dir / "run_summary.json").write_text(
        pipeline.json.dumps(
            {
                "data_root": str(Path(args.data_root).resolve()),
                "source_clips": len(manifest),
                "sample_rate": pipeline.CFG.SAMPLE_RATE,
                "feature_shape": [pipeline.CFG.N_FRAMES, pipeline.CFG.N_CHANNELS, 1],
                "n_params": int(n_params),
                "decision_threshold": float(threshold),
                "float_test_recall": float(recall),
                "float_test_fa_rate": float(false_activation_rate),
                "float_test_precision": float(precision),
                "tflite_path": str(tflite_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Training complete. Artifacts: {artifact_dir}")


if __name__ == "__main__":
    main()