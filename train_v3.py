#!/usr/bin/env python3
"""
Hey Wakey — local TinyML keyword-spotting training pipeline.

This script is based on KWS_MODEL_v2_corrected (2).ipynb, extended with a
robustness-first evaluation pass (see the CHANGES note below).

Pipeline:
  1. Discover local recordings/<speaker>/<category>/*.wav
  2. Optionally download Speech Commands negatives
  3. Optionally generate multi-voice TTS positives
  4. Leave-speaker-out train/validation/test split
  5. Audio preprocessing + augmentation
  6. TensorFlow TFLM-compatible audio microfrontend: 49 x 40
  7. DS-CNN training with validation-AUC checkpointing
  8. Validation threshold sweep + bootstrap confidence interval +
     threshold-sensitivity check
  9. Held-out test evaluation
 10. Full-int8 TFLite export
 11. Quantized-model verification
 12. C/C++ byte-array export
 13. Optional synthetic continuous-audio stream evaluation
     (event-level recall + false activations/hour)
 14. Optional leave-one-real-speaker-out CV, now with a single pooled
     decision threshold in addition to per-fold thresholds

CHANGES vs. train_v2_4.py (evaluation-robustness pass):
  - select_validation_threshold() now also returns the raw validation
    probabilities so they can be reused for bootstrapping.
  - New: bootstrap_metrics() resamples the validation split with
    replacement to report a 95% CI on recall/precision/FA-rate, so a
    single point estimate from ~70 positives is no longer taken at
    face value.
  - New: threshold_sensitivity_report() shows recall/precision/FA at
    threshold +/- 0.05 so the chosen operating point's fragility is
    visible.
  - run_leave_one_speaker_cv() now pools validation scores across all
    folds to pick ONE global threshold (what would actually ship on
    the device), and reports fold-level recall/FA under that pooled
    threshold alongside the original per-fold-tuned-threshold numbers.
  - New: a synthetic continuous-audio evaluation harness
    (build_continuous_stream / run_sliding_window_inference /
    score_continuous_stream / run_continuous_evaluation) that splices
    your existing test-split clips into a long stream with known
    keyword timestamps, runs sliding-window inference, and reports
    event-level recall and false activations per hour — the metric
    your architecture document promises but nothing before measured.

Expected local dataset:
    data/kws_dataset/
        recordings/
            speaker_1/
                positive/*.wav
                hard_negative/*.wav
                general_speech/*.wav
                noise/*.wav
            speaker_2/
                ...
        synthetic_positive/        # generated automatically if enabled

Run:
    python train_v2_5.py --data-root ./data/kws_dataset

Useful options:
    --data-root PATH
    --artifacts PATH
    --epochs N
    --skip-speech-commands
    --skip-tts
    --tts-per-voice N               # default: 300
    --run-cv                        # leave-one-speaker-out CV, with pooled threshold
    --run-continuous-eval           # synthetic continuous-audio stream eval
    --continuous-eval-minutes N     # default: 10
    --bootstrap-samples N           # default: 2000
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from tqdm.auto import tqdm

try:
    from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op
except Exception as exc:
    raise ImportError(
        "TensorFlow's audio_microfrontend_op could not be imported.\n"
        "Install the requirements in requirements/requirements.txt and use a "
        "TensorFlow version/environment that provides "
        "tensorflow.lite.experimental.microfrontend."
    ) from exc


# ============================================================
# Reproducibility
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Configuration
# ============================================================

class CFG:
    DATA_ROOT = "./data/kws_dataset"
    RECORDINGS_ROOT = f"{DATA_ROOT}/recordings"

    # Maximum synthetic positives used for training. Existing files only.
    SYNTHETIC_TRAIN_MAX = 1000
    SYNTHETIC_PER_VOICE_TRAIN_MAX = 100
    SYNTHETIC_ROOT = f"{DATA_ROOT}/synthetic_positive"

    USE_SPEECH_COMMANDS = True

    ARTIFACT_DIR = "./artifacts"

    SAMPLE_RATE = 16000
    CLIP_SECONDS = 1.0
    CLIP_LEN = int(SAMPLE_RATE * CLIP_SECONDS)

    # TFLM microfrontend parameters from the notebook.
    FRAME_LEN_MS = 30
    FRAME_HOP_MS = 20
    N_CHANNELS = 40
    N_FRAMES = 49
    LOWER_BAND_HZ = 125.0
    UPPER_BAND_HZ = 7500.0

    MICRO_ENABLE_PCAN = True
    MICRO_PCAN_STRENGTH = 0.95
    MICRO_PCAN_OFFSET = 80.0
    MICRO_ENABLE_LOG = True

    TIME_SHIFT_MS = 100
    NOISE_SNR_RANGE = (0, 20)
    PITCH_RANGE = (-1.0, 1.0)
    SPEED_RANGE = (0.9, 1.1)

    GAIN_DB_RANGE = (-6, 6)
    EQ_TILT_RANGE = (-3, 3)

    VAL_FRAC = 0.15
    TEST_FRAC = 0.15

    BATCH_SIZE = 64
    EPOCHS = 60
    LR = 1e-3

    POS_AUG_COPIES = 6
    NEG_AUG_COPIES = 2

    SPECAUG_FREQ_MASK_MAX = 6
    SPECAUG_TIME_MASK_MAX = 8
    SPECAUG_N_FREQ_MASKS = 2
    SPECAUG_N_TIME_MASKS = 2
    SPECAUG_PROB = 0.8

    GENERATE_SYNTHETIC_POSITIVES = False
    SYNTH_PER_VOICE = 300
    SYNTH_RATE_RANGE = (-15, 15)
    SYNTH_PITCH_RANGE = (-80, 80)

    SYNTH_VOICES = [
        "en-US-AriaNeural",
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
        "en-GB-LibbyNeural",
        "en-AU-NatashaNeural",
        "en-AU-WilliamNeural",
        "en-IN-NeerjaNeural",
        "en-IN-PrabhatNeural",
    ]


def configure_from_args(args):
    CFG.DATA_ROOT = str(Path(args.data_root).expanduser().resolve())
    CFG.RECORDINGS_ROOT = str(Path(CFG.DATA_ROOT) / "recordings")
    CFG.SYNTHETIC_ROOT = str(Path(CFG.DATA_ROOT) / "synthetic_positive")
    CFG.ARTIFACT_DIR = str(Path(args.artifacts).expanduser().resolve())

    CFG.EPOCHS = args.epochs
    CFG.USE_SPEECH_COMMANDS = not args.skip_speech_commands
    CFG.GENERATE_SYNTHETIC_POSITIVES = not args.skip_tts
    CFG.SYNTH_PER_VOICE = args.tts_per_voice

    Path(CFG.ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
    Path(CFG.DATA_ROOT).mkdir(parents=True, exist_ok=True)


# ============================================================
# Optional Speech Commands negatives
# ============================================================

def maybe_download_speech_commands():
    sc_dir = Path(CFG.DATA_ROOT) / "speech_commands_v2"

    if not CFG.USE_SPEECH_COMMANDS:
        return None

    wavs = list(sc_dir.rglob("*.wav")) if sc_dir.is_dir() else []
    if wavs:
        return str(sc_dir)

    sc_dir.mkdir(parents=True, exist_ok=True)
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    tar_path = sc_dir.parent / "speech_commands_v2.tar.gz"

    print("Downloading Google Speech Commands v0.02 (~2.3GB)...")
    print("If you do not want this download, use --skip-speech-commands.")

    import urllib.request
    urllib.request.urlretrieve(url, tar_path)

    import tarfile
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(sc_dir)

    print("Speech Commands download/extraction complete.")
    return str(sc_dir)


# ============================================================
# Multi-voice TTS positives
# ============================================================

def generate_tts_dataset():
    if not CFG.GENERATE_SYNTHETIC_POSITIVES:
        return

    try:
        import edge_tts
    except ImportError:
        print("edge-tts is not installed; skipping TTS generation.")
        return

    synthetic_root = Path(CFG.SYNTHETIC_ROOT)
    synthetic_root.mkdir(parents=True, exist_ok=True)

    voices = CFG.SYNTH_VOICES
    # Keep the wake phrase dominant while adding natural speaking variants.
    # The first five are the closest to the exact wake word; the remaining
    # phrases add realistic trailing context without changing the wake phrase.
    texts = [
        "Hey Wakey",
        "Hey Wakey!",
        "Hey Wakey?",
        "Hey, Wakey",
        "Hey... Wakey",
        "Hey Wakey please",
        "Hey Wakey now",
        "Hey Wakey, wake up",
        "Hey Wakey, are you there",
        "Hey Wakey, come here",
        "Hey Wakey, listen",
        "Hey Wakey, can you hear me",
    ]
    # Wider TTS variation for better speaker/style robustness.
    # 300 samples x each working voice gives ~3,000 synthetic positives
    # when the 10 currently working voices are available.
    rates = [
        "-15%", "-10%", "-5%", "+0%", "+5%", "+10%", "+15%"
    ]
    pitches = [
        "-40Hz", "-30Hz", "-20Hz", "-10Hz", "+0Hz",
        "+10Hz", "+20Hz", "+30Hz", "+40Hz"
    ]

    print("=" * 60)
    print("ROBUST TTS DATASET GENERATION")
    print("=" * 60)
    print(f"Dataset root       : {CFG.DATA_ROOT}")
    print(f"Output root        : {synthetic_root}")
    print(f"Candidate voices   : {len(voices)}")
    print(f"Samples / voice    : {CFG.SYNTH_PER_VOICE}")
    print(f"Max synthetic total: {CFG.SYNTH_PER_VOICE * len(voices)}")
    print("=" * 60)

    async def test_voice(voice):
        test_file = synthetic_root / f"_test_{voice.replace('-', '_')}.mp3"
        try:
            communicate = edge_tts.Communicate(
                text="Hey Wakey",
                voice=voice
            )
            await communicate.save(str(test_file))
            if test_file.exists() and test_file.stat().st_size > 1000:
                test_file.unlink()
                return True
        except Exception as exc:
            print(f"   Test failed: {voice}: {str(exc)[:160]}")
        if test_file.exists():
            test_file.unlink()
        return False

    async def generate_one(voice, text, rate, pitch, output_file):
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                pitch=pitch,
            )
            await communicate.save(str(output_file))
            if output_file.exists() and output_file.stat().st_size > 1000:
                return True
        except Exception:
            pass

        if output_file.exists():
            output_file.unlink()
        return False

    async def run():
        working_voices = []

        print("\nTesting voices...\n")
        for voice in voices:
            ok = await test_voice(voice)
            if ok:
                print(f"✓ {voice}")
                working_voices.append(voice)
            else:
                print(f"✗ {voice} — skipped")
            await asyncio.sleep(0.2)

        print("\n" + "=" * 60)
        print(f"Working voices: {len(working_voices)}/{len(voices)}")
        print("=" * 60)

        if not working_voices:
            print("No TTS voices are available in this environment.")
            return

        total_generated = 0

        for voice in working_voices:
            voice_dir = synthetic_root / voice
            voice_dir.mkdir(parents=True, exist_ok=True)

            generated = 0
            failed = 0

            print(f"\nGenerating: {voice}")

            for i in range(CFG.SYNTH_PER_VOICE):
                text = random.choice(texts)
                rate = random.choice(rates)
                pitch = random.choice(pitches)

                filename = (
                    f"{voice.replace('-', '_')}_"
                    f"{i:04d}_"
                    f"r{rate.replace('%', '')}_"
                    f"p{pitch.replace('Hz', '')}.mp3"
                )
                output_file = voice_dir / filename

                if output_file.exists() and output_file.stat().st_size > 1000:
                    generated += 1
                    total_generated += 1
                    continue

                success = False
                for _ in range(3):
                    success = await generate_one(
                        voice, text, rate, pitch, output_file
                    )
                    if success:
                        break
                    await asyncio.sleep(1.0)

                if success:
                    generated += 1
                    total_generated += 1
                else:
                    failed += 1

                await asyncio.sleep(0.15)

            print(f"   {voice}: {generated} generated, {failed} failed")

        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        print(f"Working voices : {len(working_voices)}")
        print(f"Total generated: {total_generated}")
        print(f"Output folder  : {synthetic_root}")
        print("=" * 60)

    asyncio.run(run())


# ============================================================
# Manifest building
# ============================================================

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".m4a", ".mp3")

CATEGORY_MAP = {
    "positive":       {"label": 1, "hard": False},
    "hard_negative":  {"label": 0, "hard": True},
    "general_speech": {"label": 0, "hard": False},
    "noise":          {"label": 0, "hard": False},
}


def fine_subgroup(path):
    stem = Path(path).stem
    m = re.match(r"^(.*)_\d+$", stem)
    return m.group(1) if m else stem


def collect_recordings(recordings_root):
    """Collect recordings without changing the user's dataset layout.

    Supported layouts:
      1. recordings/<speaker>/<category>/*.wav   (original layout)
      2. recordings/<category>/<speaker>/*.wav   (category-first layout)
      3. <data-root>/<speaker>/<category>/*.wav  (if recordings/ is absent)

    The returned manifest always uses the speaker name as `group`, so the
    existing leave-speaker-out split and all downstream training logic remain
    unchanged.
    """
    entries = []
    root = Path(recordings_root)

    # If the configured recordings/ directory is not present, use DATA_ROOT
    # only when it contains category/speaker data. This fixes path/layout
    # discovery without moving or renaming any user's files.
    if not root.is_dir():
        data_root = Path(CFG.DATA_ROOT)
        if data_root.is_dir():
            root = data_root
        else:
            print(
                f"WARNING: {recordings_root} does not exist — "
                "check --data-root."
            )
            return entries

    seen = set()

    def add_file(path, speaker, category):
        path = str(Path(path).resolve())
        key = (path, speaker, category)
        if key in seen:
            return
        seen.add(key)
        meta = CATEGORY_MAP[category]
        entries.append({
            "path": path,
            "label": meta["label"],
            "hard": meta["hard"],
            "group": speaker,
            "fine_group": f"{speaker}__{fine_subgroup(path)}",
            "category": category,
        })

    # Layout 1: <root>/<speaker>/<category>/*
    for speaker_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        speaker = speaker_dir.name
        if speaker in {"speech_commands_v2", "synthetic_positive", "artifacts"}:
            continue
        for category in CATEGORY_MAP:
            cat_dir = speaker_dir / category
            if not cat_dir.is_dir():
                continue
            for path in cat_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
                    add_file(path, speaker, category)

    # Layout 2: <root>/<category>/<speaker>/*
    for category in CATEGORY_MAP:
        cat_root = root / category
        if not cat_root.is_dir():
            continue
        for speaker_dir in sorted(p for p in cat_root.iterdir() if p.is_dir()):
            speaker = speaker_dir.name
            for path in speaker_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
                    add_file(path, speaker, category)

    return entries


def build_manifest(sc_dir):
    manifest = collect_recordings(CFG.RECORDINGS_ROOT)

    noise_files = [
        e["path"]
        for e in manifest
        if e["category"] == "noise"
    ]

    sc_neg_files = []

    if sc_dir:
        for wav in glob.glob(f"{sc_dir}/**/*.wav", recursive=True):
            if "_background_noise_" in wav:
                noise_files.append(wav)
            else:
                sc_neg_files.append(wav)

        random.shuffle(sc_neg_files)
        sc_neg_files = sc_neg_files[:4000]

        manifest.extend([
            {
                "path": p,
                "label": 0,
                "hard": False,
                "group": f"sc__{Path(p).parent.name}",
                "fine_group": f"sc__{Path(p).parent.name}",
                "category": "speech_commands",
            }
            for p in sc_neg_files
        ])

    by_cat = defaultdict(int)
    for e in manifest:
        by_cat[e["category"]] += 1

    print("\nManifest counts by category:")
    for cat, n in by_cat.items():
        print(f"  {cat:18s}: {n}")

    print(f"Total manifest entries: {len(manifest)}")

    real_speakers = sorted({
        e["group"]
        for e in manifest
        if not e["group"].startswith("sc__")
    })
    print(f"Speakers found: {real_speakers}")

    n_pos = sum(e["label"] == 1 for e in manifest)
    n_neg = sum(e["label"] == 0 for e in manifest)

    if n_pos == 0:
        raise RuntimeError(
            "No positive 'Hey Wakey' clips found. "
            "Expected recordings/<speaker>/positive/*.wav"
        )
    if n_neg == 0:
        raise RuntimeError(
            "No negative clips found. "
            "Expected hard_negative/general_speech/noise data."
        )

    return manifest, noise_files


# ============================================================
# Leave-speaker-out split
# ============================================================

def create_main_split(manifest):
    labels = np.array([m["label"] for m in manifest])

    real_entries = [
        i for i, m in enumerate(manifest)
        if not m["group"].startswith("sc__")
    ]
    sc_entries = [
        i for i, m in enumerate(manifest)
        if m["group"].startswith("sc__")
    ]

    real_speakers = sorted(set(manifest[i]["group"] for i in real_entries))
    n_real = len(real_speakers)

    pos_per_speaker = defaultdict(int)
    for i in real_entries:
        if manifest[i]["label"] == 1:
            pos_per_speaker[manifest[i]["group"]] += 1

    print(f"\nReal speakers found: {n_real}")
    print(f"Positive clips per speaker: {dict(pos_per_speaker)}")

    if n_real < 3:
        raise RuntimeError(
            "At least 3 real speakers are needed for the notebook's "
            "train/validation/test leave-speaker-out split."
        )

    rng = random.Random(SEED)
    shuffled_speakers = real_speakers[:]
    rng.shuffle(shuffled_speakers)

    n_test_spk = max(1, round(n_real * CFG.TEST_FRAC))
    n_val_spk = max(1, round(n_real * CFG.VAL_FRAC))

    if n_test_spk + n_val_spk >= n_real:
        n_test_spk = 1
        n_val_spk = 1 if n_real >= 3 else 0

    test_speakers = set(shuffled_speakers[:n_test_spk])
    val_speakers = set(
        shuffled_speakers[n_test_spk:n_test_spk + n_val_spk]
    )
    train_speakers = (
        set(shuffled_speakers)
        - test_speakers
        - val_speakers
    )

    print(f"train speakers: {sorted(train_speakers)}")
    print(f"val speakers:   {sorted(val_speakers)}")
    print(f"test speakers:  {sorted(test_speakers)}")

    if n_real < 6:
        print(
            "WARNING: Small speaker count. This is leave-speaker-out "
            "evaluation, not a random sample split."
        )

    real_train_idx = [
        i for i in real_entries
        if manifest[i]["group"] in train_speakers
    ]
    real_val_idx = [
        i for i in real_entries
        if manifest[i]["group"] in val_speakers
    ]
    real_test_idx = [
        i for i in real_entries
        if manifest[i]["group"] in test_speakers
    ]

    if sc_entries:
        sc_groups = np.array(
            [manifest[i]["group"] for i in sc_entries]
        )
        sc_idx_arr = np.array(sc_entries)

        gss1 = GroupShuffleSplit(
            n_splits=1,
            test_size=CFG.TEST_FRAC,
            random_state=SEED,
        )
        sc_trainval, sc_test = next(
            gss1.split(sc_idx_arr, groups=sc_groups)
        )

        gss2 = GroupShuffleSplit(
            n_splits=1,
            test_size=CFG.VAL_FRAC / (1 - CFG.TEST_FRAC),
            random_state=SEED,
        )
        sc_train, sc_val = next(
            gss2.split(
                sc_idx_arr[sc_trainval],
                groups=sc_groups[sc_trainval],
            )
        )

        sc_train_idx = sc_idx_arr[sc_trainval][sc_train].tolist()
        sc_val_idx = sc_idx_arr[sc_trainval][sc_val].tolist()
        sc_test_idx = sc_idx_arr[sc_test].tolist()
    else:
        sc_train_idx, sc_val_idx, sc_test_idx = [], [], []

    train_idx = np.array(real_train_idx + sc_train_idx)
    val_idx = np.array(real_val_idx + sc_val_idx)
    test_idx = np.array(real_test_idx + sc_test_idx)

    groups = np.array([m["group"] for m in manifest])

    print(
        f"\nSplit strategy: leave-speaker-out "
        f"({n_test_spk} test, {n_val_spk} val, "
        f"{n_real - n_test_spk - n_val_spk} train speakers)"
    )

    for name, ids in {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }.items():
        n_pos = int(labels[ids].sum())
        print(
            f"{name:5s}: {len(ids):5d} clips "
            f"({n_pos} positive, {len(ids)-n_pos} negative, "
            f"{len(set(groups[ids]))} groups)"
        )

    assert not (train_speakers & val_speakers)
    assert not (train_speakers & test_speakers)
    assert not (val_speakers & test_speakers)

    if labels[val_idx].sum() == 0:
        raise RuntimeError("Validation split has zero positives.")
    if labels[test_idx].sum() == 0:
        raise RuntimeError("Test split has zero positives.")

    print("Speaker leakage check passed.")

    return train_idx, val_idx, test_idx


# ============================================================
# Audio preprocessing + augmentation
# ============================================================

def load_audio(path, sr=CFG.SAMPLE_RATE):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def fit_to_length(y, length=CFG.CLIP_LEN):
    if len(y) < length:
        pad = length - len(y)
        left = pad // 2
        y = np.pad(y, (left, pad - left))
    elif len(y) > length:
        start = (len(y) - length) // 2
        y = y[start:start + length]
    return y


def peak_normalize(y, eps=1e-9):
    peak = np.max(np.abs(y))
    if peak < eps:
        return y
    return y / peak * 0.98


def random_time_shift(y, max_ms=CFG.TIME_SHIFT_MS):
    max_shift = int(CFG.SAMPLE_RATE * max_ms / 1000)
    shift = random.randint(-max_shift, max_shift)
    return np.roll(y, shift)


def random_gain(y, db_range=CFG.GAIN_DB_RANGE):
    db = random.uniform(*db_range)
    return y * (10 ** (db / 20))


def random_eq_tilt(y, tilt_range=CFG.EQ_TILT_RANGE):
    tilt_db_per_oct = random.uniform(*tilt_range)
    n = len(y)

    freqs = np.fft.rfftfreq(
        n,
        d=1.0 / CFG.SAMPLE_RATE
    )
    freqs[0] = 1.0

    octaves = np.log2(freqs / 1000.0)
    gain_db = tilt_db_per_oct * octaves
    gain = 10 ** (gain_db / 20)

    spec = np.fft.rfft(y)
    spec *= gain

    return np.fft.irfft(spec, n=n).astype(np.float32)


def mix_noise(y, noise_pool, snr_range=CFG.NOISE_SNR_RANGE):
    if not noise_pool:
        return y

    noise_path = random.choice(noise_pool)
    noise, _ = librosa.load(
        noise_path,
        sr=CFG.SAMPLE_RATE,
        mono=True
    )

    if len(noise) == 0:
        return y

    if len(noise) < len(y):
        reps = int(np.ceil(len(y) / len(noise)))
        noise = np.tile(noise, reps)

    start = random.randint(0, len(noise) - len(y))
    noise = noise[start:start + len(y)]

    sig_power = np.mean(y ** 2) + 1e-9
    noise_power = np.mean(noise ** 2) + 1e-9

    snr_db = random.uniform(*snr_range)
    target_noise_power = sig_power / (10 ** (snr_db / 10))

    noise = noise * np.sqrt(target_noise_power / noise_power)
    return y + noise


def random_pitch_speed(
    y,
    pitch_range=CFG.PITCH_RANGE,
    speed_range=CFG.SPEED_RANGE,
):
    if random.random() < 0.5:
        n_steps = random.uniform(*pitch_range)
        y = librosa.effects.pitch_shift(
            y,
            sr=CFG.SAMPLE_RATE,
            n_steps=n_steps,
        )

    if random.random() < 0.5:
        rate = random.uniform(*speed_range)
        y = librosa.effects.time_stretch(y, rate=rate)

    return y


def augment(y, noise_pool):
    y = random_pitch_speed(y)
    y = fit_to_length(y)
    y = random_time_shift(y)
    y = random_gain(y)

    if random.random() < 0.7:
        y = random_eq_tilt(y)

    if random.random() < 0.8:
        y = mix_noise(y, noise_pool)

    y = peak_normalize(y)
    return y


def preprocess_clean(y):
    y = fit_to_length(y)
    y = peak_normalize(y)
    return y


# ============================================================
# TFLM microfrontend feature extraction
# ============================================================

def extract_microfrontend(y):
    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)

    pcm16 = np.round(y * 32767.0).astype(np.int16)

    features = audio_microfrontend_op.audio_microfrontend(
        tf.convert_to_tensor(pcm16, dtype=tf.int16),
        sample_rate=CFG.SAMPLE_RATE,
        window_size=CFG.FRAME_LEN_MS,
        window_step=CFG.FRAME_HOP_MS,
        num_channels=CFG.N_CHANNELS,
        upper_band_limit=CFG.UPPER_BAND_HZ,
        lower_band_limit=CFG.LOWER_BAND_HZ,
        enable_pcan=CFG.MICRO_ENABLE_PCAN,
        pcan_strength=CFG.MICRO_PCAN_STRENGTH,
        pcan_offset=CFG.MICRO_PCAN_OFFSET,
        enable_log=CFG.MICRO_ENABLE_LOG,
        out_type=tf.float32,
    ).numpy()

    if features.shape[0] < CFG.N_FRAMES:
        features = np.pad(
            features,
            ((0, CFG.N_FRAMES - features.shape[0]), (0, 0)),
            mode="edge",
        )
    elif features.shape[0] > CFG.N_FRAMES:
        features = features[:CFG.N_FRAMES]

    if features.shape[1] != CFG.N_CHANNELS:
        raise ValueError(
            f"Unexpected microfrontend width {features.shape[1]} "
            f"(expected {CFG.N_CHANNELS})."
        )

    return features.astype(np.float32)


def spec_augment(
    features,
    freq_mask_max=CFG.SPECAUG_FREQ_MASK_MAX,
    time_mask_max=CFG.SPECAUG_TIME_MASK_MAX,
    n_freq_masks=CFG.SPECAUG_N_FREQ_MASKS,
    n_time_masks=CFG.SPECAUG_N_TIME_MASKS,
):
    features = features.copy()
    n_frames, n_channels = features.shape
    mask_value = features.mean()

    for _ in range(n_freq_masks):
        f = random.randint(
            0,
            min(freq_mask_max, n_channels)
        )
        if f:
            f0 = random.randint(0, n_channels - f)
            features[:, f0:f0 + f] = mask_value

    for _ in range(n_time_masks):
        t = random.randint(
            0,
            min(time_mask_max, n_frames)
        )
        if t:
            t0 = random.randint(0, n_frames - t)
            features[t0:t0 + t, :] = mask_value

    return features


# ============================================================
# Build feature datasets
# ============================================================

def build_split(entries, noise_pool, is_train, aug_copies_pos, aug_copies_neg):
    X = []
    y_labels = []

    synthetic_loaded = 0
    synthetic_failed = 0

    for e in tqdm(
        entries,
        desc="train" if is_train else "eval"
    ):
        try:
            raw = load_audio(e["path"])
        except Exception as exc:
            print(f"skip {e['path']}: {exc}")

            if e.get("category") == "synthetic_positive":
                synthetic_failed += 1

            continue

        if e.get("category") == "synthetic_positive":
            synthetic_loaded += 1

        clean = preprocess_clean(raw.copy())

        X.append(extract_microfrontend(clean))
        y_labels.append(e["label"])

        if is_train:
            n_copies = (
                aug_copies_pos
                if e["label"] == 1
                else aug_copies_neg
            )

            for _ in range(n_copies):
                try:
                    aug_y = augment(raw.copy(), noise_pool)
                    feat = extract_microfrontend(aug_y)

                    if random.random() < CFG.SPECAUG_PROB:
                        feat = spec_augment(feat)

                    X.append(feat)
                    y_labels.append(e["label"])
                except Exception:
                    continue

    if not X:
        raise RuntimeError("No features were generated.")

    X = np.stack(X)
    y_labels = np.array(y_labels, dtype=np.int32)

    if is_train:
        print("\nSynthetic positive loading:")
        print(f"  Successfully loaded : {synthetic_loaded}")
        print(f"  Failed to load      : {synthetic_failed}")

    return X, y_labels


def build_feature_datasets(manifest, noise_files, train_idx, val_idx, test_idx):
    train_entries = [manifest[i] for i in train_idx]
    val_entries = [manifest[i] for i in val_idx]
    test_entries = [manifest[i] for i in test_idx]

    synthetic_root = Path(CFG.SYNTHETIC_ROOT)
    synthetic_by_voice = defaultdict(list)
    for p in sorted(synthetic_root.rglob("*.mp3")):
        synthetic_by_voice[p.parent.name].append(p)

    synthetic_positive_files = []
    for voice in sorted(synthetic_by_voice):
        synthetic_positive_files.extend(
            synthetic_by_voice[voice][:CFG.SYNTHETIC_PER_VOICE_TRAIN_MAX]
        )

    synthetic_positive_files = synthetic_positive_files[:CFG.SYNTHETIC_TRAIN_MAX]

    print(
        f"\nFound {sum(len(v) for v in synthetic_by_voice.values())} synthetic "
        "positive audio files on disk."
    )
    print(
        f"Using {len(synthetic_positive_files)} synthetic positive files "
        f"for training (cap={CFG.SYNTHETIC_TRAIN_MAX}, "
        f"per-voice={CFG.SYNTHETIC_PER_VOICE_TRAIN_MAX})."
    )

    for p in synthetic_positive_files:
        train_entries.append({
            "path": str(p),
            "label": 1,
            "hard": False,
            "group": f"tts__{p.parent.name}",
            "fine_group": f"tts__{p.stem}",
            "category": "synthetic_positive",
        })

    print(
        "Training source clips after synthetic positives: "
        f"{len(train_entries)} "
        f"(+{len(synthetic_positive_files)} synthetic positives)"
    )

    X_train, y_train = build_split(
        train_entries,
        noise_files,
        True,
        CFG.POS_AUG_COPIES,
        CFG.NEG_AUG_COPIES,
    )

    X_val, y_val = build_split(
        val_entries,
        noise_files,
        False,
        0,
        0,
    )

    X_test, y_test = build_split(
        test_entries,
        noise_files,
        False,
        0,
        0,
    )

    print("\n" + "=" * 60)
    print("FEATURE DATASET")
    print("=" * 60)
    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")
    print(
        f"\nTrain class balance -- "
        f"pos: {y_train.sum()}, "
        f"neg: {(y_train == 0).sum()}"
    )

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    assert X_train.shape[1:] == (
        CFG.N_FRAMES,
        CFG.N_CHANNELS,
        1,
    )
    assert X_val.shape[1:] == (
        CFG.N_FRAMES,
        CFG.N_CHANNELS,
        1,
    )
    assert X_test.shape[1:] == (
        CFG.N_FRAMES,
        CFG.N_CHANNELS,
        1,
    )

    cache_path = Path(CFG.ARTIFACT_DIR) / "features_cache.npz"
    np.savez_compressed(
        cache_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    print(f"Feature cache saved: {cache_path}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================
# DS-CNN model
# ============================================================

def build_ds_cnn(
    input_shape=(CFG.N_FRAMES, CFG.N_CHANNELS, 1),
    n_ds_blocks=3,
    channels=64,
):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        10,
        kernel_size=(4, 10),
        strides=(2, 1),
        padding="same",
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    for _ in range(n_ds_blocks):
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            channels,
            kernel_size=1,
            padding="same",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(
        2,
        activation="softmax"
    )(x)

    return tf.keras.Model(
        inputs,
        outputs,
        name="hey_wakey_ds_cnn",
    )


def compile_model(model, learning_rate=None):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate or CFG.LR
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="acc"),
            tf.keras.metrics.Precision(
                name="precision",
                class_id=1,
            ),
            tf.keras.metrics.Recall(
                name="recall",
                class_id=1,
            ),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


# ============================================================
# Train final model
# ============================================================

def train_final_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    train_mean = X_train.mean()
    train_std = X_train.std() + 1e-9

    X_train_n = (X_train - train_mean) / train_std
    X_val_n = (X_val - train_mean) / train_std
    X_test_n = (X_test - train_mean) / train_std

    np.save(
        Path(CFG.ARTIFACT_DIR) / "norm_mean.npy",
        train_mean,
    )
    np.save(
        Path(CFG.ARTIFACT_DIR) / "norm_std.npy",
        train_std,
    )

    n_pos = y_train.sum()
    n_neg = (y_train == 0).sum()
    total = n_pos + n_neg

    class_weight = {
        0: (total / (2.0 * n_neg)) * 1.3,
        1: (total / (2.0 * n_pos)),
    }

    print("Class weights:", class_weight)

    y_train_oh = tf.keras.utils.to_categorical(y_train, 2)
    y_val_oh = tf.keras.utils.to_categorical(y_val, 2)

    model = build_ds_cnn()
    model = compile_model(model)

    model.summary()
    n_params = model.count_params()
    print(f"Total params: {n_params}")

    artifact = Path(CFG.ARTIFACT_DIR)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=10,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(artifact / "best_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train_n,
        y_train_oh,
        validation_data=(X_val_n, y_val_oh),
        epochs=CFG.EPOCHS,
        batch_size=CFG.BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(artifact / "hey_wakey_float.keras")

    best_model_path = artifact / "best_model.keras"
    if best_model_path.exists():
        model = tf.keras.models.load_model(best_model_path)
        print("Reloaded best validation-AUC checkpoint:", best_model_path)

    # Save training history.
    hist_path = artifact / "training_history.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                k: [float(v) for v in vals]
                for k, vals in history.history.items()
            },
            f,
            indent=2,
        )

    print("\nDataset check:")
    print(f"Validation: total={len(y_val)}, "
          f"positive={np.sum(y_val == 1)}, "
          f"negative={np.sum(y_val == 0)}")
    print(f"Test:       total={len(y_test)}, "
          f"positive={np.sum(y_test == 1)}, "
          f"negative={np.sum(y_test == 0)}")

    return (
        model,
        X_train_n,
        X_val_n,
        X_test_n,
        train_mean,
        train_std,
        n_params,
    )


# ============================================================
# Threshold sweep
# ============================================================

def threshold_metrics(y_true, probs, thr):
    preds = (probs >= thr).astype(int)

    tp = np.sum((preds == 1) & (y_true == 1))
    tn = np.sum((preds == 0) & (y_true == 0))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))

    recall = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    fa_rate = fp / (fp + tn + 1e-9)

    return {
        "threshold": float(thr),
        "recall": float(recall),
        "precision": float(precision),
        "fa_rate": float(fa_rate),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def select_validation_threshold(model, X_val_n, y_val):
    val_probs = model.predict(
        X_val_n,
        batch_size=256,
        verbose=0,
    )[:, 1]

    n_val_pos = int(y_val.sum())
    n_val_neg = int((y_val == 0).sum())

    print(
        f"\nValidation set: "
        f"{n_val_pos} positive, {n_val_neg} negative clips."
    )

    sweep = [
        threshold_metrics(y_val, val_probs, thr)
        for thr in np.linspace(0.01, 0.99, 99)
    ]

    print("\n" + "=" * 60)
    print("VALIDATION THRESHOLD TRADE-OFF")
    print("=" * 60)

    for target in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        feasible = [
            s for s in sweep
            if s["fa_rate"] <= target
        ]

        if feasible:
            best = max(feasible, key=lambda s: s["recall"])
            print(
                f"FA <= {target:5.1%} : "
                f"threshold={best['threshold']:.2f}, "
                f"recall={best['recall']:.3f}, "
                f"precision={best['precision']:.3f}, "
                f"FA={best['fa_rate']:.4f}"
            )
        else:
            print(
                f"FA <= {target:5.1%} : no threshold found"
            )

    target_fa_rate = 0.05
    feasible = [
        s for s in sweep
        if s["fa_rate"] <= target_fa_rate
    ]

    if feasible:
        chosen = max(feasible, key=lambda s: s["recall"])
    else:
        chosen = max(
            sweep,
            key=lambda s: s["recall"] - s["fa_rate"],
        )

    decision_threshold = chosen["threshold"]

    print("\n" + "=" * 60)
    print("SELECTED VALIDATION OPERATING POINT")
    print("=" * 60)
    print(f"Threshold : {decision_threshold:.3f}")
    print(f"Recall    : {chosen['recall']:.4f}")
    print(f"Precision : {chosen['precision']:.4f}")
    print(f"FA rate   : {chosen['fa_rate']:.4f}")
    print(
        "\nWARNING: This threshold is selected using validation "
        "data only. It is not a final deployment threshold until "
        "checked on held-out speakers / continuous audio."
    )

    return decision_threshold, sweep, val_probs


# ============================================================
# Validation robustness diagnostics (NEW)
# ============================================================

def nearest_sweep_point(sweep, target_threshold):
    """Return the sweep entry whose threshold is closest to target_threshold."""
    return min(sweep, key=lambda s: abs(s["threshold"] - target_threshold))


def threshold_sensitivity_report(sweep, center_threshold, delta=0.05):
    """Show recall/precision/FA at center_threshold +/- delta.

    A large gap between 'lower' and 'upper' means the chosen operating
    point sits on a knife's edge -- small threshold drift (e.g. from
    quantization, from a slightly different microphone, from recalibrating
    later) could swing your reported numbers a lot.
    """
    lower = nearest_sweep_point(sweep, center_threshold - delta)
    center = nearest_sweep_point(sweep, center_threshold)
    upper = nearest_sweep_point(sweep, center_threshold + delta)

    print("\n" + "=" * 60)
    print(f"THRESHOLD SENSITIVITY (center +/- {delta:.2f})")
    print("=" * 60)
    for label, point in [("lower", lower), ("center", center), ("upper", upper)]:
        print(
            f"  {label:6s} thr={point['threshold']:.3f}  "
            f"recall={point['recall']:.3f}  "
            f"precision={point['precision']:.3f}  "
            f"FA={point['fa_rate']:.4f}"
        )

    return {"lower": lower, "center": center, "upper": upper}


def bootstrap_metrics(y_true, probs, threshold, n_bootstrap=2000, seed=SEED):
    """Bootstrap recall/precision/FA-rate at a fixed threshold.

    This quantifies how much a small validation set can make these numbers
    swing purely from which clips happened to be included. Positives and
    negatives are resampled separately (with replacement) so every
    resample keeps both classes represented whenever the original split
    does.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    recalls, precisions, fa_rates = [], [], []

    for _ in range(n_bootstrap):
        bp = (
            rng.choice(pos_idx, size=len(pos_idx), replace=True)
            if len(pos_idx) else np.array([], dtype=int)
        )
        bn = (
            rng.choice(neg_idx, size=len(neg_idx), replace=True)
            if len(neg_idx) else np.array([], dtype=int)
        )
        idx = np.concatenate([bp, bn])

        m = threshold_metrics(y_true[idx], probs[idx], threshold)
        recalls.append(m["recall"])
        precisions.append(m["precision"])
        fa_rates.append(m["fa_rate"])

    def summarize(values):
        arr = np.array(values)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p2_5": float(np.percentile(arr, 2.5)),
            "p97_5": float(np.percentile(arr, 97.5)),
        }

    report = {
        "threshold": float(threshold),
        "n_bootstrap": n_bootstrap,
        "n_positive": int(len(pos_idx)),
        "n_negative": int(len(neg_idx)),
        "recall": summarize(recalls),
        "precision": summarize(precisions),
        "fa_rate": summarize(fa_rates),
    }

    print("\n" + "=" * 60)
    print(
        f"VALIDATION BOOTSTRAP ({n_bootstrap} resamples, "
        f"{len(pos_idx)} positives / {len(neg_idx)} negatives)"
    )
    print("=" * 60)
    for k in ["recall", "precision", "fa_rate"]:
        s = report[k]
        print(
            f"  {k:10s}: mean={s['mean']:.3f}  "
            f"95% CI=[{s['p2_5']:.3f}, {s['p97_5']:.3f}]"
        )
    print(
        "\nA wide CI here means your validation set is too small to trust "
        "a single point estimate -- treat recall/FA numbers from this split "
        "as a range, not a fact, until cross-validation or more data narrows it."
    )

    return report


# ============================================================
# Float test evaluation
# ============================================================

def evaluate_float_model(
    model,
    X_test_n,
    y_test,
    decision_threshold,
    sweep,
    n_params,
):
    test_probs = model.predict(
        X_test_n,
        batch_size=256,
        verbose=0,
    )[:, 1]

    test_preds = (
        test_probs >= decision_threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        test_preds,
        labels=[0, 1],
    )
    tn, fp, fn, tp = cm.ravel()

    test_recall = tp / (tp + fn + 1e-9)
    test_fa_rate = fp / (fp + tn + 1e-9)
    test_precision = tp / (tp + fp + 1e-9)

    print("\nConfusion matrix [ [TN FP] [FN TP] ]:")
    print(cm)
    print(f"Test recall:    {test_recall:.4f}")
    print(f"Test FA rate:   {test_fa_rate:.4f}")
    print(f"Test precision: {test_precision:.4f}")

    report = {
        "decision_threshold": float(decision_threshold),
        "test_confusion_matrix": cm.tolist(),
        "test_recall": float(test_recall),
        "test_fa_rate": float(test_fa_rate),
        "test_precision": float(test_precision),
        "pr_sweep": sweep,
        "n_params": int(n_params),
    }

    with open(
        Path(CFG.ARTIFACT_DIR) / "eval_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    return test_recall, test_fa_rate, test_precision


# ============================================================
# Firmware feature contract
# ============================================================

def save_firmware_contract(train_mean, train_std):
    contract = {
        "sample_rate": CFG.SAMPLE_RATE,
        "window_size_ms": CFG.FRAME_LEN_MS,
        "window_step_ms": CFG.FRAME_HOP_MS,
        "num_channels": CFG.N_CHANNELS,
        "lower_band_limit_hz": CFG.LOWER_BAND_HZ,
        "upper_band_limit_hz": CFG.UPPER_BAND_HZ,
        "enable_pcan": CFG.MICRO_ENABLE_PCAN,
        "pcan_strength": CFG.MICRO_PCAN_STRENGTH,
        "pcan_offset": CFG.MICRO_PCAN_OFFSET,
        "enable_log": CFG.MICRO_ENABLE_LOG,
        "feature_shape": [CFG.N_FRAMES, CFG.N_CHANNELS],
        "normalization_mean": float(train_mean),
        "normalization_std": float(train_std),
    }

    with open(
        Path(CFG.ARTIFACT_DIR) / "firmware_feature_contract.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(contract, f, indent=2)

    print("\n=== ESP32/TFLM MICROFRONTEND CONTRACT ===")
    for k, v in contract.items():
        print(f"{k:24s}: {v}")


# ============================================================
# Full INT8 TFLite export
# ============================================================

def export_int8_tflite(model, X_train_n):
    def representative_dataset():
        n = min(300, len(X_train_n))
        idxs = np.random.choice(
            len(X_train_n),
            n,
            replace=False,
        )
        for i in idxs:
            sample = X_train_n[i:i + 1].astype(np.float32)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = (
        Path(CFG.ARTIFACT_DIR) /
        "hey_wakey_int8.tflite"
    )

    tflite_path.write_bytes(tflite_model)

    print(
        f"Quantized model size: "
        f"{len(tflite_model) / 1024:.1f} KB"
    )

    return tflite_path


# ============================================================
# Quantized model verification
# ============================================================

def evaluate_quantized_model(
    tflite_path,
    X_test_n,
    y_test,
    decision_threshold,
    float_recall,
    float_fa_rate,
):
    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path)
    )
    interpreter.allocate_tensors()

    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    in_scale, in_zero = in_detail["quantization"]
    out_scale, out_zero = out_detail["quantization"]

    if in_scale == 0:
        raise RuntimeError(
            "Quantized input scale is zero."
        )

    def quantize_input(x_float):
        return np.clip(
            np.round(
                x_float / in_scale + in_zero
            ),
            -128,
            127,
        ).astype(np.int8)

    quant_probs = []

    for i in tqdm(
        range(len(X_test_n)),
        desc="Quantized test",
    ):
        x_q = quantize_input(
            X_test_n[i:i + 1]
        )

        interpreter.set_tensor(
            in_detail["index"],
            x_q,
        )
        interpreter.invoke()

        out_q = interpreter.get_tensor(
            out_detail["index"]
        )[0]

        out_f = (
            out_q.astype(np.float32) - out_zero
        ) * out_scale

        quant_probs.append(out_f[1])

    quant_probs = np.array(
        quant_probs,
        dtype=np.float32,
    )

    quant_preds = (
        quant_probs >= decision_threshold
    ).astype(int)

    cm_q = confusion_matrix(
        y_test,
        quant_preds,
        labels=[0, 1],
    )
    tn_q, fp_q, fn_q, tp_q = cm_q.ravel()

    quant_recall = tp_q / (tp_q + fn_q + 1e-9)
    quant_fa_rate = fp_q / (fp_q + tn_q + 1e-9)

    print("\nQuantized model test confusion matrix:")
    print(cm_q)
    print(
        f"Quantized recall:  {quant_recall:.4f} "
        f"(float was {float_recall:.4f})"
    )
    print(
        f"Quantized FA rate: {quant_fa_rate:.4f} "
        f"(float was {float_fa_rate:.4f})"
    )

    q_report = {
        "input_quantization": {
            "scale": float(in_scale),
            "zero_point": int(in_zero),
        },
        "output_quantization": {
            "scale": float(out_scale),
            "zero_point": int(out_zero),
        },
        "confusion_matrix": cm_q.tolist(),
        "recall": float(quant_recall),
        "fa_rate": float(quant_fa_rate),
        "float_recall": float(float_recall),
        "float_fa_rate": float(float_fa_rate),
    }

    with open(
        Path(CFG.ARTIFACT_DIR) /
        "quantized_eval_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(q_report, f, indent=2)


# ============================================================
# Export TFLite bytes to C/C++
# ============================================================

def export_c_array(tflite_path):
    data = Path(tflite_path).read_bytes()

    cpp_path = (
        Path(CFG.ARTIFACT_DIR) /
        "model_data.cc"
    )
    header_path = (
        Path(CFG.ARTIFACT_DIR) /
        "model_data.h"
    )

    # Avoid requiring xxd, so this works on Windows/Linux/macOS.
    lines = []
    for i in range(0, len(data), 12):
        chunk = data[i:i + 12]
        lines.append(
            "    " +
            ", ".join(f"0x{b:02x}" for b in chunk) +
            ","
        )

    cpp = (
        '#include "model_data.h"\n\n'
        "const unsigned char g_hey_wakey_model_data[] = {\n"
        + "\n".join(lines)
        + "\n};\n\n"
        f"const unsigned int g_hey_wakey_model_data_len = {len(data)};\n"
    )

    header = (
        "#ifndef HEY_WAKEY_MODEL_DATA_H_\n"
        "#define HEY_WAKEY_MODEL_DATA_H_\n\n"
        "extern const unsigned char g_hey_wakey_model_data[];\n"
        "extern const unsigned int g_hey_wakey_model_data_len;\n\n"
        "#endif  // HEY_WAKEY_MODEL_DATA_H_\n"
    )

    cpp_path.write_text(cpp, encoding="utf-8")
    header_path.write_text(header, encoding="utf-8")

    print(
        f"Firmware C array written to "
        f"{cpp_path} / {header_path}"
    )


# ============================================================
# Leave-one-speaker-out CV
# ============================================================

CV_EPOCHS = 40
CV_AUG_COPIES_POS = 4
CV_AUG_COPIES_NEG = 2
CV_VAL_FRAC = 0.12


def sweep_threshold_cv(
    y_true,
    probs,
    target_fa=0.05,
    min_recall=0.5,
):
    sweep = []

    for thr in np.linspace(0.01, 0.99, 99):
        preds = (probs >= thr).astype(int)

        neg_mask = y_true == 0
        pos_mask = y_true == 1

        fa = (
            preds[neg_mask].mean()
            if neg_mask.sum()
            else 0.0
        )
        rec = (
            preds[pos_mask].mean()
            if pos_mask.sum()
            else 0.0
        )

        sweep.append({
            "threshold": float(thr),
            "fa_rate": float(fa),
            "recall": float(rec),
        })

    feasible = [
        s for s in sweep
        if s["fa_rate"] <= target_fa
    ]

    best = (
        max(feasible, key=lambda s: s["recall"])
        if feasible else None
    )

    if best is None or best["recall"] < min_recall:
        relaxed = [
            s for s in sweep
            if s["fa_rate"] <= target_fa * 2
        ]

        best = (
            max(relaxed, key=lambda s: s["recall"])
            if relaxed
            else max(
                sweep,
                key=lambda s:
                    s["recall"] - s["fa_rate"],
            )
        )

    return best


def run_leave_one_speaker_cv(manifest, noise_files):
    real_speakers = sorted({
        m["group"]
        for m in manifest
        if not m["group"].startswith("sc__")
    })

    cv_results = []

    # NEW: keep raw per-fold scores so we can pool them into one global
    # threshold afterward, instead of only ever picking a threshold from
    # each fold's own (small) validation split.
    fold_val_probs = []
    fold_val_labels = []
    fold_test_probs = []
    fold_test_labels = []

    for fold_i, held_speaker in enumerate(real_speakers):
        print(
            f"\n{'=' * 70}\n"
            f"FOLD {fold_i + 1}/{len(real_speakers)} "
            f"-- held-out speaker: {held_speaker}\n"
            f"{'=' * 70}"
        )

        test_entries_cv = [
            m for m in manifest
            if m["group"] == held_speaker
        ]
        trainpool_entries = [
            m for m in manifest
            if m["group"] != held_speaker
        ]

        trainpool_labels = np.array([
            m["label"]
            for m in trainpool_entries
        ])

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=CV_VAL_FRAC,
            random_state=SEED + fold_i,
        )

        tr_i, va_i = next(
            sss.split(
                np.arange(len(trainpool_entries)),
                trainpool_labels,
            )
        )

        fold_train_entries = [
            trainpool_entries[i]
            for i in tr_i
        ]
        fold_val_entries = [
            trainpool_entries[i]
            for i in va_i
        ]

        synthetic_by_voice = defaultdict(list)
        for p in sorted(Path(CFG.SYNTHETIC_ROOT).rglob("*.mp3")):
            synthetic_by_voice[p.parent.name].append(p)

        synthetic_positive_files = []
        for voice in sorted(synthetic_by_voice):
            synthetic_positive_files.extend(
                synthetic_by_voice[voice][:CFG.SYNTHETIC_PER_VOICE_TRAIN_MAX]
            )
        synthetic_positive_files = synthetic_positive_files[:CFG.SYNTHETIC_TRAIN_MAX]

        fold_train_entries += [
            {
                "path": str(p),
                "label": 1,
                "hard": False,
                "group": f"tts__{p.parent.name}",
                "fine_group": f"tts__{p.stem}",
                "category": "synthetic_positive",
            }
            for p in synthetic_positive_files
        ]

        print(
            f"train: {len(fold_train_entries)} source clips | "
            f"val: {len(fold_val_entries)} | "
            f"test: {len(test_entries_cv)}"
        )

        Xtr, ytr = build_split(
            fold_train_entries,
            noise_files,
            True,
            CV_AUG_COPIES_POS,
            CV_AUG_COPIES_NEG,
        )
        Xva, yva = build_split(
            fold_val_entries,
            noise_files,
            False,
            0,
            0,
        )
        Xte, yte = build_split(
            test_entries_cv,
            noise_files,
            False,
            0,
            0,
        )

        Xtr = Xtr[..., np.newaxis]
        Xva = Xva[..., np.newaxis]
        Xte = Xte[..., np.newaxis]

        fold_mean = Xtr.mean()
        fold_std = Xtr.std() + 1e-9

        Xtr_n = (Xtr - fold_mean) / fold_std
        Xva_n = (Xva - fold_mean) / fold_std
        Xte_n = (Xte - fold_mean) / fold_std

        n_pos_f = ytr.sum()
        n_neg_f = (ytr == 0).sum()
        total_f = n_pos_f + n_neg_f

        cw = {
            0: (total_f / (2.0 * n_neg_f)) * 1.3,
            1: (total_f / (2.0 * n_pos_f)),
        }

        fold_model = build_ds_cnn()
        compile_model(fold_model)

        fold_cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=8,
                restore_best_weights=True,
            )
        ]

        fold_model.fit(
            Xtr_n,
            tf.keras.utils.to_categorical(ytr, 2),
            validation_data=(
                Xva_n,
                tf.keras.utils.to_categorical(yva, 2),
            ),
            epochs=CV_EPOCHS,
            batch_size=CFG.BATCH_SIZE,
            class_weight=cw,
            callbacks=fold_cb,
            verbose=0,
        )

        val_probs_f = fold_model.predict(
            Xva_n,
            batch_size=256,
            verbose=0,
        )[:, 1]

        thr_info = sweep_threshold_cv(
            yva,
            val_probs_f,
            target_fa=0.05,
            min_recall=0.5,
        )

        test_probs_f = fold_model.predict(
            Xte_n,
            batch_size=256,
            verbose=0,
        )[:, 1]

        test_preds_f = (
            test_probs_f >= thr_info["threshold"]
        ).astype(int)

        cm_f = confusion_matrix(
            yte,
            test_preds_f,
            labels=[0, 1],
        )

        tn_f, fp_f, fn_f, tp_f = cm_f.ravel()

        fold_recall = tp_f / (
            tp_f + fn_f + 1e-9
        )
        fold_fa = fp_f / (
            fp_f + tn_f + 1e-9
        )

        print(
            f"chosen threshold: {thr_info['threshold']:.2f}"
        )
        print(
            f"HELD-OUT SPEAKER TEST -- "
            f"recall: {fold_recall:.3f}  "
            f"FA rate: {fold_fa:.4f} "
            f"(TN={tn_f} FP={fp_f} "
            f"FN={fn_f} TP={tp_f})"
        )

        cv_results.append({
            "held_speaker": held_speaker,
            "threshold": thr_info["threshold"],
            "recall": float(fold_recall),
            "fa_rate": float(fold_fa),
            "n_test_pos": int(yte.sum()),
            "n_test_neg": int((yte == 0).sum()),
        })

        # NEW: retain raw scores for pooled-threshold analysis below.
        fold_val_probs.append(val_probs_f)
        fold_val_labels.append(yva)
        fold_test_probs.append(test_probs_f)
        fold_test_labels.append(yte)

        tf.keras.backend.clear_session()

    print(
        f"\n{'=' * 70}\n"
        "LEAVE-ONE-SPEAKER-OUT CV SUMMARY "
        "(each fold's own threshold, tuned on its own small val split)\n"
        f"{'=' * 70}"
    )

    for r in cv_results:
        print(
            f"{r['held_speaker']:15s} "
            f"recall={r['recall']:.3f} "
            f"FA={r['fa_rate']:.4f} "
            f"thr={r['threshold']:.2f}"
        )

    mean_recall = float(
        np.mean([r["recall"] for r in cv_results])
    )
    std_recall = float(
        np.std([r["recall"] for r in cv_results])
    )
    mean_fa = float(
        np.mean([r["fa_rate"] for r in cv_results])
    )

    print(
        f"\nMean recall across speakers: "
        f"{mean_recall:.3f} (+/- {std_recall:.3f})"
    )
    print(
        f"Mean FA rate across speakers: "
        f"{mean_fa:.4f}"
    )

    # ------------------------------------------------------------
    # NEW: pooled-threshold analysis.
    #
    # Each fold above picked its own threshold from its own tiny
    # validation split, which is itself a small-sample-noise problem --
    # exactly the thing this pass is trying to fix. A device in the
    # field ships with ONE threshold, not a different one per speaker.
    # So: pool every fold's validation scores into one set, pick a
    # single threshold from that, and re-score every fold's held-out
    # test speaker with that same threshold.
    # ------------------------------------------------------------
    pooled_val_probs = np.concatenate(fold_val_probs)
    pooled_val_labels = np.concatenate(fold_val_labels)

    pooled_thr_info = sweep_threshold_cv(
        pooled_val_labels,
        pooled_val_probs,
        target_fa=0.05,
        min_recall=0.5,
    )
    pooled_threshold = pooled_thr_info["threshold"]

    print(
        f"\nPooled validation threshold (all folds combined): "
        f"{pooled_threshold:.3f}"
    )
    print(
        "This is a single threshold that would actually ship on the "
        "device, rather than a different threshold per speaker."
    )

    pooled_fold_results = []
    for held_speaker, probs_i, labels_i in zip(
        real_speakers, fold_test_probs, fold_test_labels
    ):
        preds_i = (probs_i >= pooled_threshold).astype(int)
        cm_i = confusion_matrix(labels_i, preds_i, labels=[0, 1])
        tn_i, fp_i, fn_i, tp_i = cm_i.ravel()

        recall_i = tp_i / (tp_i + fn_i + 1e-9)
        fa_i = fp_i / (fp_i + tn_i + 1e-9)

        pooled_fold_results.append({
            "held_speaker": held_speaker,
            "recall": float(recall_i),
            "fa_rate": float(fa_i),
        })

    mean_recall_pooled = float(
        np.mean([r["recall"] for r in pooled_fold_results])
    )
    std_recall_pooled = float(
        np.std([r["recall"] for r in pooled_fold_results])
    )
    mean_fa_pooled = float(
        np.mean([r["fa_rate"] for r in pooled_fold_results])
    )

    print("\n" + "=" * 70)
    print("RESULTS WITH A SINGLE POOLED THRESHOLD (more realistic for ESP32)")
    print("=" * 70)
    for r in pooled_fold_results:
        print(
            f"{r['held_speaker']:15s} "
            f"recall={r['recall']:.3f} "
            f"FA={r['fa_rate']:.4f}"
        )
    print(
        f"\nMean recall (pooled threshold): "
        f"{mean_recall_pooled:.3f} (+/- {std_recall_pooled:.3f})"
    )
    print(
        f"Mean FA rate (pooled threshold): {mean_fa_pooled:.4f}"
    )
    print(
        "\nCompare this against the per-fold-tuned-threshold numbers above: "
        "a big gap between them means your per-fold thresholds were "
        "overfit to each fold's own small validation split, and the "
        "pooled-threshold numbers are the more honest estimate of what "
        "you'd actually see on the device."
    )

    with open(
        Path(CFG.ARTIFACT_DIR) / "kfold_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "per_fold_threshold_results": cv_results,
                "mean_recall_per_fold_threshold": mean_recall,
                "std_recall_per_fold_threshold": std_recall,
                "mean_fa_rate_per_fold_threshold": mean_fa,
                "pooled_threshold": pooled_threshold,
                "pooled_threshold_results": pooled_fold_results,
                "mean_recall_pooled_threshold": mean_recall_pooled,
                "std_recall_pooled_threshold": std_recall_pooled,
                "mean_fa_rate_pooled_threshold": mean_fa_pooled,
            },
            f,
            indent=2,
        )


# ============================================================
# Continuous-audio evaluation (NEW)
# ============================================================
#
# Everything above evaluates isolated 1-second clips. The real ESP32 use
# case is continuous listening, where the model sees a never-ending
# stream and the metric that actually matters is "false activations per
# hour" plus "did it catch the keyword when it was actually said". This
# section builds an approximate continuous stream out of your existing
# test-split clips (so it needs no new recordings), runs the model over
# it with a sliding window, and scores at the event level.

def build_continuous_stream(
    pos_paths,
    neg_paths,
    target_minutes=10,
    min_gap_s=3.0,
    max_gap_s=15.0,
    seed=SEED,
):
    """Splice existing clips into one long stream with known keyword times.

    Returns (stream, events) where `stream` is a 1-D float32 waveform and
    `events` is a list of (start_sample, end_sample) tuples marking where
    a positive "Hey Wakey" clip was inserted.
    """
    rng = random.Random(seed)
    sr = CFG.SAMPLE_RATE
    target_samples = int(target_minutes * 60 * sr)

    pos_paths = list(pos_paths)
    neg_paths = list(neg_paths)
    if not pos_paths or not neg_paths:
        raise RuntimeError(
            "Need at least one positive and one negative clip to build "
            "a continuous stream."
        )

    rng.shuffle(pos_paths)
    rng.shuffle(neg_paths)

    pos_cycle = itertools.cycle(pos_paths)
    neg_cycle = itertools.cycle(neg_paths)

    chunks = []
    events = []
    cur = 0

    while cur < target_samples:
        # Negative "gap" audio between keyword events.
        gap_target = int(rng.uniform(min_gap_s, max_gap_s) * sr)
        gap_accum = 0
        while gap_accum < gap_target and cur < target_samples:
            neg_path = next(neg_cycle)
            try:
                y = peak_normalize(load_audio(neg_path))
            except Exception:
                continue
            chunks.append(y)
            gap_accum += len(y)
            cur += len(y)

        if cur >= target_samples:
            break

        # One positive keyword event, kept at its natural (clean) length
        # rather than forced into the fixed 1s training window, so the
        # sliding-window inference below has to actually find it.
        pos_path = next(pos_cycle)
        try:
            y = peak_normalize(load_audio(pos_path))
        except Exception:
            continue

        start = cur
        chunks.append(y)
        cur += len(y)
        events.append((start, cur))

    stream = np.concatenate(chunks)[:target_samples].astype(np.float32)
    events = [(s, min(e, target_samples)) for s, e in events if s < target_samples]

    return stream, events


def run_sliding_window_inference(
    model,
    stream,
    train_mean,
    train_std,
    hop_ms=200,
    window_s=1.0,
    batch_size=256,
):
    """Slide a 1s window over `stream` and return (timestamps, probs)."""
    sr = CFG.SAMPLE_RATE
    hop = max(1, int(sr * hop_ms / 1000))
    win = int(sr * window_s)
    n = len(stream)

    starts = list(range(0, max(1, n - win + 1), hop))
    if not starts:
        starts = [0]

    timestamps = []
    windows = []

    for start in starts:
        seg = stream[start:start + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        seg = peak_normalize(seg)
        windows.append(extract_microfrontend(seg))
        timestamps.append(start / sr)

    X = np.stack(windows)[..., np.newaxis]
    X_n = (X - train_mean) / train_std

    probs = model.predict(X_n, batch_size=batch_size, verbose=0)[:, 1]

    return np.array(timestamps), probs


def score_continuous_stream(
    timestamps,
    probs,
    events,
    threshold,
    tolerance_s=0.5,
    refractory_s=1.0,
):
    """Event-level scoring: a detection counts once per event, within
    `tolerance_s` of the event's boundaries, and detections closer than
    `refractory_s` to the previous accepted detection are merged so one
    utterance can't be double-counted as multiple false activations.
    """
    sr = CFG.SAMPLE_RATE
    event_windows = [(s / sr, e / sr) for s, e in events]

    fired = timestamps[probs >= threshold]

    accepted = []
    last = -1e9
    for t in fired:
        if t - last >= refractory_s:
            accepted.append(t)
            last = t

    matched_events = set()
    true_positive_detections = 0

    for d in accepted:
        hit_idx = None
        for i, (s, e) in enumerate(event_windows):
            if i in matched_events:
                continue
            if (s - tolerance_s) <= d <= (e + tolerance_s):
                hit_idx = i
                break
        if hit_idx is not None:
            matched_events.add(hit_idx)
            true_positive_detections += 1

    false_activations = len(accepted) - true_positive_detections
    event_recall = (
        len(matched_events) / max(1, len(event_windows))
    )
    duration_hours = (
        (timestamps[-1] + 1.0) / 3600.0 if len(timestamps) else 0.0
    )
    fa_per_hour = (
        false_activations / duration_hours
        if duration_hours > 0 else float("nan")
    )

    return {
        "threshold": float(threshold),
        "n_events": len(event_windows),
        "n_detected_events": len(matched_events),
        "event_recall": float(event_recall),
        "n_detections_total": len(accepted),
        "false_activations": int(false_activations),
        "stream_duration_hours": float(duration_hours),
        "false_activations_per_hour": float(fa_per_hour),
    }


def run_continuous_evaluation(
    model,
    manifest,
    test_idx,
    train_mean,
    train_std,
    decision_threshold,
    minutes=10,
):
    """Build a synthetic continuous stream from the TEST split only (no
    train/val leakage), run sliding-window inference, and report
    event-level recall + false activations/hour.
    """
    test_entries = [manifest[i] for i in test_idx]
    pos_paths = [e["path"] for e in test_entries if e["label"] == 1]
    neg_paths = [e["path"] for e in test_entries if e["label"] == 0]

    print("\n" + "=" * 60)
    print("CONTINUOUS-AUDIO EVALUATION")
    print("=" * 60)

    if not pos_paths:
        print("Skipped: no positive clips in the test split.")
        return None
    if not neg_paths:
        print("Skipped: no negative clips in the test split.")
        return None

    print(
        f"Building a {minutes}-minute synthetic stream from "
        f"{len(pos_paths)} test positives and {len(neg_paths)} test negatives..."
    )

    stream, events = build_continuous_stream(
        pos_paths, neg_paths, target_minutes=minutes
    )

    print(
        f"Stream length: {len(stream) / CFG.SAMPLE_RATE / 60:.1f} min, "
        f"{len(events)} embedded keyword events."
    )

    timestamps, probs = run_sliding_window_inference(
        model, stream, train_mean, train_std
    )

    result = score_continuous_stream(
        timestamps, probs, events, decision_threshold
    )

    print(
        f"Event recall           : {result['event_recall']:.3f} "
        f"({result['n_detected_events']}/{result['n_events']})"
    )
    print(f"False activations      : {result['false_activations']}")
    print(f"Stream duration        : {result['stream_duration_hours']:.3f} hours")
    print(f"False activations/hour : {result['false_activations_per_hour']:.2f}")
    print(
        "\nThis stream is spliced from your existing clips, so it's an "
        "approximation of continuous listening, not a real-room recording -- "
        "still a much closer proxy to deployment than isolated 1s clips, "
        "but a real recorded test (ideally through the INMP441/ESP32 path) "
        "should replace it before final submission."
    )

    with open(
        Path(CFG.ARTIFACT_DIR) / "continuous_eval_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Hey Wakey TinyML KWS model locally."
    )

    parser.add_argument(
        "--data-root",
        default="./data/kws_dataset",
        help="Dataset root containing recordings/.",
    )
    parser.add_argument(
        "--artifacts",
        default="./artifacts",
        help="Output directory for models/reports.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Final-model training epochs.",
    )
    parser.add_argument(
        "--skip-speech-commands",
        action="store_true",
        help="Do not download/use Speech Commands negatives.",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Do not generate TTS positives.",
    )
    parser.add_argument(
    "--tts-per-voice",
    type=int,
    default=300,   
    help="Synthetic clips per working TTS voice.",
    )
    parser.add_argument(
        "--run-cv",
        action="store_true",
        help=(
            "Run leave-one-speaker-out CV after final export, including "
            "the pooled-threshold analysis."
        ),
    )
    parser.add_argument(
        "--run-continuous-eval",
        action="store_true",
        help=(
            "Build a synthetic continuous-audio stream from the test "
            "split and report event-level recall + false activations/hour."
        ),
    )
    parser.add_argument(
        "--continuous-eval-minutes",
        type=int,
        default=10,
        help="Length in minutes of the synthetic continuous-audio stream.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for the validation CI report.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    configure_from_args(args)

    print("=" * 70)
    print("HEY WAKEY — LOCAL TINYML KWS TRAINING")
    print("=" * 70)
    print(f"Dataset root : {CFG.DATA_ROOT}")
    print(f"Artifacts    : {CFG.ARTIFACT_DIR}")
    print(f"TensorFlow   : {tf.__version__}")
    print(
        "GPU devices  :",
        tf.config.list_physical_devices("GPU")
    )
    print("=" * 70)

    # Optional data preparation.
    sc_dir = maybe_download_speech_commands()
    generate_tts_dataset()

    # Manifest + split.
    manifest, noise_files = build_manifest(sc_dir)
    train_idx, val_idx, test_idx = create_main_split(manifest)

    # Feature extraction.
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = build_feature_datasets(
        manifest,
        noise_files,
        train_idx,
        val_idx,
        test_idx,
    )

    # Model training.
    (
        model,
        X_train_n,
        X_val_n,
        X_test_n,
        train_mean,
        train_std,
        n_params,
    ) = train_final_model(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    # Threshold selection + float evaluation.
    decision_threshold, sweep, val_probs = (
        select_validation_threshold(
            model,
            X_val_n,
            y_val,
        )
    )

    # NEW: quantify how much the small validation set can swing these
    # numbers, and how sensitive the chosen threshold is.
    bootstrap_report = bootstrap_metrics(
        y_val,
        val_probs,
        decision_threshold,
        n_bootstrap=args.bootstrap_samples,
    )
    with open(
        Path(CFG.ARTIFACT_DIR) / "validation_bootstrap_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(bootstrap_report, f, indent=2)

    sensitivity_report = threshold_sensitivity_report(sweep, decision_threshold)
    with open(
        Path(CFG.ARTIFACT_DIR) / "threshold_sensitivity_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(sensitivity_report, f, indent=2)

    (
        float_recall,
        float_fa_rate,
        float_precision,
    ) = evaluate_float_model(
        model,
        X_test_n,
        y_test,
        decision_threshold,
        sweep,
        n_params,
    )

    # Firmware contract.
    save_firmware_contract(
        train_mean,
        train_std,
    )

    # Full INT8 export.
    tflite_path = export_int8_tflite(
        model,
        X_train_n,
    )

    # Quantized verification.
    evaluate_quantized_model(
        tflite_path,
        X_test_n,
        y_test,
        decision_threshold,
        float_recall,
        float_fa_rate,
    )

    # C/C++ export.
    export_c_array(tflite_path)

    # NEW: synthetic continuous-audio evaluation (opt-in, since it adds
    # runtime and needs the test split to have both classes).
    if args.run_continuous_eval:
        run_continuous_evaluation(
            model,
            manifest,
            test_idx,
            train_mean,
            train_std,
            decision_threshold,
            minutes=args.continuous_eval_minutes,
        )

    # Save a final run summary.
    with open(
        Path(CFG.ARTIFACT_DIR) / "run_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "sample_rate": CFG.SAMPLE_RATE,
                "feature_shape": [
                    CFG.N_FRAMES,
                    CFG.N_CHANNELS,
                    1,
                ],
                "n_params": int(n_params),
                "decision_threshold": float(
                    decision_threshold
                ),
                "float_test_recall": float(
                    float_recall
                ),
                "float_test_fa_rate": float(
                    float_fa_rate
                ),
                "float_test_precision": float(
                    float_precision
                ),
                "tflite_path": str(tflite_path),
            },
            f,
            indent=2,
        )

    if args.run_cv:
        run_leave_one_speaker_cv(
            manifest,
            noise_files,
        )

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Artifacts: {CFG.ARTIFACT_DIR}")
    print("Important files:")
    for name in [
        "best_model.keras",
        "hey_wakey_float.keras",
        "hey_wakey_int8.tflite",
        "model_data.cc",
        "model_data.h",
        "norm_mean.npy",
        "norm_std.npy",
        "eval_report.json",
        "validation_bootstrap_report.json",
        "threshold_sensitivity_report.json",
        "quantized_eval_report.json",
        "continuous_eval_report.json",
        "firmware_feature_contract.json",
        "kfold_report.json",
        "run_summary.json",
    ]:
        p = Path(CFG.ARTIFACT_DIR) / name
        print(f"  {'OK' if p.exists() else '--'}  {p}")
    print("=" * 70)


if __name__ == "__main__":
    main()