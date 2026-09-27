#!/usr/bin/env python3
"""
"Hey Wakey" KWS dataset recorder
2 s clips, 16 kHz, mono, 16 bit PCM WAV, recorded from the laptop microphone.

Setup
    pip install sounddevice numpy scipy

Commands
    python record_dataset.py check                      set the mic level first
    python record_dataset.py speaker [--pos-scale 1.5]  one person per session (~130 clips, ~12 min)
    python record_dataset.py ambient --label tv_speech --count 100
    python record_dataset.py splits                     speaker independent 70/15/15
    python record_dataset.py --list-devices

Folder layout (created automatically)
    kws_dataset/
      raw/
        positive/hey_wakey/spk001/*.wav
        negative/hard_negative/spk001/*.wav      similar sounding words, other assistant wake words
        negative/speech_other/spk001/*.wav       ordinary short phrases
        negative/non_speech/spk001/*.wav         silence, typing, cough, clap, laugh ...
        negative/background/<label>/*.wav        TV, fan, AC, street ... (no speaker)
      external/speech_commands/                  drop the public corpus negatives here
      external/background_noise/                 drop extra noise clips here
      processed/mfcc/                            later: .npy features per split
      metadata/speakers.csv  clips.csv  splits.csv
"""
import argparse
import csv
import random
import sys
import time
import wave
from datetime import datetime
from math import gcd
from pathlib import Path

import numpy as np

SR = 16000
DUR = 2.0
N = int(SR * DUR)

ROOT = Path("kws_dataset")
RAW = ROOT / "raw"
META = ROOT / "metadata"
SPEAKERS_CSV = META / "speakers.csv"
CLIPS_CSV = META / "clips.csv"
SPLITS_CSV = META / "splits.csv"

DIRS = [
    "raw/positive/hey_wakey",
    "raw/negative/hard_negative",
    "raw/negative/speech_other",
    "raw/negative/non_speech",
    "raw/negative/background",
    "external/speech_commands",
    "external/background_noise",
    "processed/mfcc",
    "metadata",
]

SPK_FIELDS = ["speaker_id", "gender", "age_group", "native_language", "consent", "created_at"]
CLIP_FIELDS = ["file", "speaker_id", "class", "label", "text", "block", "distance_cm", "style",
               "peak_dbfs", "snr_db", "speech_start_s", "speech_end_s", "qc_ok", "recorded_at"]

# ----------------------------------------------------------------------------
# What gets recorded for each speaker
# ----------------------------------------------------------------------------
STYLES = {
    "normal": "normal voice, natural pace",
    "fast": "say it quickly",
    "slow": "say it slowly, stretch the words",
    "loud": "loud, like calling across a room",
    "soft": "soft, quiet voice (not a whisper)",
    "question": "as a question, rising tone",
    "excited": "excited, high energy",
    "flat": "flat monotone, bored",
    "stress_hey": "stress the first word: HEY wakey",
    "stress_wa": "stress the second word: hey WA-key",
}

HARD = {  # sounds close to the keyword, or other assistants' wake words
    "hey_becky": "Hey Becky", "hey_whiskey": "Hey Whiskey", "hey_wacky": "Hey Wacky",
    "hey_lucky": "Hey Lucky", "hey_rocky": "Hey Rocky", "hey_vicky": "Hey Vicky",
    "hey_siri": "Hey Siri", "okay_google": "Okay Google", "alexa": "Alexa",
    "wake_up": "Wake up", "wakey_wakey": "Wakey wakey", "hey_alone": "Hey",
    "wakey_alone": "Wakey", "hey_there": "Hey there", "hey_bakery": "Hey bakery",
    "hey_kiki": "Hey Kiki",
}
HARD_MID = ["hey_becky", "hey_whiskey", "hey_wacky", "hey_lucky", "hey_rocky",
            "hey_vicky", "hey_siri", "okay_google", "wake_up", "hey_kiki"]
HARD_FAR = ["hey_becky", "hey_whiskey", "hey_wacky", "okay_google", "alexa", "wake_up"]

SPEECH_POOL = [
    "What time is it", "Turn on the light", "Good morning everyone", "How are you doing",
    "Can you hear me", "Let's get started", "See you tomorrow", "Open the door",
    "I'll be right back", "That sounds good", "Play some music", "Thank you very much",
    "Where are my keys", "Turn the volume down", "What's the weather", "Call me later",
    "I am almost ready", "One two three four", "Nice to meet you", "Please close the window",
    "Are you coming", "Set an alarm", "Good night", "Have a great day", "It is so hot today",
    "Let me check", "I don't know", "Just a moment", "Come here quickly",
    "The meeting starts soon", "Send me the file", "How much is this", "Take a left here",
    "I'm hungry", "What is your name", "Let's go home", "Look at that", "Stop right there",
    "Bring some water", "Read the message",
]

NON_SPEECH = [  # (tag, instruction, repeats)
    ("silence", "stay completely quiet, room tone only", 3),
    ("typing", "type on the laptop keyboard normally", 2),
    ("cough", "cough once or twice", 2),
    ("clap", "clap your hands twice", 2),
    ("laugh", "laugh naturally", 1),
    ("throat_clear", "clear your throat", 1),
    ("desk_tap", "tap the desk a few times", 1),
]

BLOCKS = {
    "near": (30, "NEAR: about 30 cm from the laptop (one forearm away), facing the screen."),
    "mid": (100, "MID: about 1 m from the laptop. Sit back or stand, facing the screen."),
    "angled": (100, "ANGLED: still about 1 m away, but turn your face about 45 degrees away from the laptop."),
    "far": (200, "FAR: about 2 m from the laptop, facing it. Speak so the laptop can still hear you."),
}


def build_prompts(spk, pos_scale=1.0):
    """130 prompts per speaker at pos_scale 1.0: 70 positive, 32 hard negative,
    16 speech, 12 non speech. pos_scale multiplies every positive count."""
    sid = f"spk{spk:03d}"
    rng = random.Random(spk)
    P = []

    def add(block, cls, tag, text, how, style=""):
        P.append(dict(sid=sid, block=block, cls=cls, tag=tag, text=text, how=how,
                      dist=BLOCKS[block][0], style=style, label=int(cls.startswith("positive"))))

    def pos(block, plan):
        for style, n in plan:
            for k in range(1, max(1, round(n * pos_scale)) + 1):
                add(block, "positive/hey_wakey", f"pos_{block}_{style}_{k:02d}", "Hey Wakey", STYLES[style], style)

    def hard(block, keys):
        for key in keys:
            add(block, "negative/hard_negative", f"hard_{key}_{block}", HARD[key], "normal voice")

    pos("near", [("normal", 6), ("fast", 2), ("slow", 2), ("loud", 2), ("soft", 2), ("question", 2),
                 ("excited", 2), ("flat", 2), ("stress_hey", 1), ("stress_wa", 1)])
    hard("near", list(HARD))
    for j, phrase in enumerate(rng.sample(SPEECH_POOL, 16), 1):
        add("near", "negative/speech_other", f"speech_{j:02d}", phrase, "normal voice")
    for tag, how, reps in NON_SPEECH:
        for k in range(1, reps + 1):
            add("near", "negative/non_speech", f"{tag}_{k:02d}", tag, how)

    pos("mid", [("normal", 6), ("fast", 2), ("slow", 2), ("loud", 2), ("soft", 2), ("question", 2),
                ("excited", 2), ("flat", 2), ("stress_hey", 1), ("stress_wa", 1)])
    hard("mid", HARD_MID)

    pos("angled", [("normal", 4), ("fast", 2), ("loud", 2), ("question", 2), ("soft", 2)])

    pos("far", [("normal", 6), ("loud", 4), ("slow", 2), ("excited", 2)])
    hard("far", HARD_FAR)
    return P


# ----------------------------------------------------------------------------
# Audio capture and quality checks
# ----------------------------------------------------------------------------
_sd = None
_native_rate = None


def get_sd():
    global _sd
    if _sd is None:
        try:
            import sounddevice as sd
        except (ImportError, OSError) as e:
            sys.exit(f"sounddevice is not available ({e}).\nRun: pip install sounddevice numpy scipy")
        _sd = sd
    return _sd


def capture(seconds, device=None):
    """Record mono int16 at 16 kHz. Falls back to the device rate plus resampling."""
    global _native_rate
    sd = get_sd()
    n = int(seconds * SR)
    if _native_rate is None:
        try:
            x = sd.rec(n, samplerate=SR, channels=1, dtype="int16", device=device)
            sd.wait()
            return x[:, 0].copy()
        except sd.PortAudioError:
            _native_rate = int(sd.query_devices(device, "input")["default_samplerate"])
            print(f"   (device has no 16 kHz mode, recording at {_native_rate} Hz and resampling)")
    from scipy.signal import resample_poly
    x = sd.rec(int(seconds * _native_rate), samplerate=_native_rate, channels=1, dtype="float32", device=device)
    sd.wait()
    g = gcd(SR, _native_rate)
    y = resample_poly(x[:, 0], SR // g, _native_rate // g)
    y = np.clip(y * 32767, -32768, 32767).astype(np.int16)
    out = np.zeros(n, np.int16)
    out[:min(n, len(y))] = y[:n]
    return out


def write_wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(x.astype(np.int16).tobytes())


def frame_rms(f, win=400, hop=160):
    n = 1 + max(0, (len(f) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt((f[idx] ** 2).mean(axis=1) + 1e-12)


def analyse(x):
    """Peak, SNR and a rough speech onset/offset (25 ms frames, 10 ms hop)."""
    f = x.astype(np.float32) / 32768.0
    r = frame_rms(f)
    floor = float(np.percentile(r, 10)) + 1e-9
    top = float(r.max()) + 1e-9
    act = np.where(r > max(4 * floor, 0.08 * top))[0]
    start, end = (act[0] * 0.01, act[-1] * 0.01 + 0.025) if len(act) >= 3 else (None, None)
    return dict(
        peak_dbfs=20 * np.log10(float(np.abs(f).max()) + 1e-9),
        top_dbfs=20 * np.log10(top),
        snr_db=20 * np.log10(top / floor),
        clip=float((np.abs(x) >= 32000).mean()),
        start=start, end=end)


def judge(p, a):
    problems = []
    if p["cls"].endswith("non_speech"):
        if p["tag"].startswith("silence") and a["top_dbfs"] > -40:
            problems.append("the room is not quiet enough for a silence clip")
        return problems
    if a["top_dbfs"] < -45:
        problems.append("too quiet, speak up or move closer")
    elif a["snr_db"] < 6:
        problems.append("speech is not clearly above the noise")
    if a["clip"] > 0.002:
        problems.append("clipping, lower the input volume or speak softer")
    if a["start"] is None:
        problems.append("no speech detected")
    else:
        if a["start"] < 0.03:
            problems.append("started too early, wait for REC")
        if a["end"] > 1.96:
            problems.append("cut off at the end, say it a bit quicker")
    return problems


# ----------------------------------------------------------------------------
# Metadata helpers
# ----------------------------------------------------------------------------
def ensure_dirs():
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_manifest():
    return {r["file"]: r for r in read_csv(CLIPS_CSV)}


def save_manifest(m):
    write_csv(CLIPS_CSV, CLIP_FIELDS, [m[k] for k in sorted(m)])


def clip_row(rel, sid, cls, label, text, block, dist, style, a, ok):
    fmt = lambda v: "" if v is None else f"{v:.3f}"
    return {"file": rel, "speaker_id": sid, "class": cls, "label": label, "text": text,
            "block": block, "distance_cm": dist, "style": style,
            "peak_dbfs": f"{a['peak_dbfs']:.1f}", "snr_db": f"{a['snr_db']:.1f}",
            "speech_start_s": fmt(a["start"]), "speech_end_s": fmt(a["end"]),
            "qc_ok": int(ok), "recorded_at": datetime.now().isoformat(timespec="seconds")}


def pick_speaker():
    rows = read_csv(SPEAKERS_CSV)
    nums = [int(r["speaker_id"][3:]) for r in rows]
    default = max(nums, default=0) + 1
    s = input(f"Speaker number [Enter = new speaker {default}, or type a number to resume]: ").strip()
    spk = int(s) if s else default
    sid = f"spk{spk:03d}"
    if sid not in {r["speaker_id"] for r in rows}:
        print("\nConsent: this person agrees their voice recordings can be used to train and evaluate the model.")
        if input("Agreed? [y/N] ").strip().lower() != "y":
            sys.exit("No consent, nothing recorded.")
        print("Optional details for balance checks, press Enter to skip any.")
        gender = input("  Gender: ").strip()
        age = input("  Age group (e.g. 18-24): ").strip()
        lang = input("  Native language / accent: ").strip()
        rows.append(dict(speaker_id=sid, gender=gender, age_group=age, native_language=lang,
                         consent="yes", created_at=datetime.now().isoformat(timespec="seconds")))
        write_csv(SPEAKERS_CSV, SPK_FIELDS, rows)
    return spk


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def run_session(args):
    ensure_dirs()
    spk = pick_speaker()
    prompts = build_prompts(spk, getattr(args, "pos_scale", 1.0))
    manifest = load_manifest()
    total = len(prompts)
    i, force, last_block = 0, False, None

    print(f"\nSession for spk{spk:03d}: {total} clips of {DUR:.0f} s.")
    print("Speak the moment you see REC. Existing clips are skipped, so you can resume any time.\n")

    while i < total:
        p = prompts[i]
        rel = f"{p['cls']}/{p['sid']}/{p['sid']}_{p['tag']}.wav"
        path = RAW / rel
        if path.exists() and not force:
            i += 1
            continue
        force = False

        if p["block"] != last_block:
            print("\n" + "=" * 70 + f"\n{BLOCKS[p['block']][1]}\n" + "=" * 70)
            input("Ready? Press Enter ")
            last_block = p["block"]

        kind = {"positive": "POSITIVE", "hard_negative": "HARD NEG", "speech_other": "SPEECH",
                "non_speech": "NOISE"}[p["cls"].split("/")[1] if p["label"] == 0 else "positive"]
        verb = "do this" if kind == "NOISE" else "say"
        target = p["how"] if kind == "NOISE" else f"\"{p['text']}\"  ({p['how']})"
        print(f"\n[{i + 1:>2}/{total}] {kind:<8} {verb}: {target}")
        cmd = input("   Enter=record  b=back  s=skip  q=quit > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            i += 1
            continue
        if cmd == "b":
            i = max(0, i - 1)
            force = True
            continue

        while True:
            time.sleep(0.4)  # let the key click die down
            print("   >>> REC", flush=True)
            x = capture(DUR, args.device)
            a = analyse(x)
            problems = judge(p, a)
            span = "" if a["start"] is None else f"  speech {a['start']:.2f}-{a['end']:.2f} s"
            print(f"   peak {a['peak_dbfs']:.1f} dBFS  snr {a['snr_db']:.0f} dB{span}")
            if not problems:
                ok = True
                break
            print("   Problem: " + "; ".join(problems))
            if input("   Enter=retry  k=keep anyway > ").strip().lower() == "k":
                ok = False
                break

        write_wav(path, x)
        manifest[rel] = clip_row(rel, p["sid"], p["cls"], p["label"], p["text"], p["block"],
                                 p["dist"], p["style"], a, ok)
        save_manifest(manifest)
        i += 1
    else:
        print(f"\nDone. spk{spk:03d} is complete. Thank you!")
        return
    print(f"\nStopped early. Run again and enter {spk} to resume spk{spk:03d}.")


def run_ambient(args):
    ensure_dirs()
    label = args.label
    d = RAW / "negative" / "background" / label
    start = len(list(d.glob("*.wav"))) + 1 if d.exists() else 1
    print(f"Recording {args.count} clips of {DUR:.0f} s of '{label}' ({args.count * DUR / 60:.1f} min).")
    print("Start the sound source now (TV, fan, street, kitchen ...). Starting in 5 s.")
    time.sleep(5)
    x = capture(args.count * DUR, args.device)
    m = load_manifest()
    for k in range(args.count):
        clip = x[k * N:(k + 1) * N]
        rel = f"negative/background/{label}/bg_{label}_{start + k:04d}.wav"
        write_wav(RAW / rel, clip)
        m[rel] = clip_row(rel, f"env_{label}", "negative/background", 0, label, "ambient", "", "", analyse(clip), True)
    save_manifest(m)
    print(f"Saved {args.count} clips to {d}")


def run_check(args):
    dev = get_sd().query_devices(args.device, "input")
    print(f"Input device: {dev['name']}")
    print("Turn OFF mic enhancements (noise suppression, AGC, voice focus) in your OS sound settings.")
    print("Say 'Hey Wakey' from ~30 cm after each prompt. Ctrl+C to stop. Target peak: -18 to -3 dBFS.\n")
    try:
        while True:
            input("Enter to test > ")
            time.sleep(0.4)
            a = analyse(capture(DUR, args.device))
            bar = "#" * max(0, int((a["peak_dbfs"] + 60) / 2))
            verdict = ("too hot, lower the input volume" if a["peak_dbfs"] > -3
                       else "too low, raise the input volume" if a["peak_dbfs"] < -18 else "good")
            print(f"  peak {a['peak_dbfs']:.1f} dBFS  {bar}  {verdict}")
    except (KeyboardInterrupt, EOFError):
        print()


def run_splits(args):
    m = load_manifest()
    if not m:
        sys.exit("No clips.csv yet.")
    spk = sorted({r["speaker_id"] for r in m.values() if r["speaker_id"].startswith("spk")})
    random.Random(args.seed).shuffle(spk)
    n = len(spk)
    n_val = round(0.15 * n)
    n_test = round(0.15 * n)
    n_train = n - n_val - n_test
    split_of = {s: "train" if i < n_train else "val" if i < n_train + n_val else "test"
                for i, s in enumerate(spk)}

    env = {}
    for f in sorted(m):
        if m[f]["speaker_id"].startswith("env_"):
            env.setdefault(m[f]["speaker_id"], []).append(f)
    file_split = {}
    for files in env.values():  # ambient chunks: split by time order so neighbours stay together
        k = len(files)
        for j, f in enumerate(files):
            file_split[f] = "train" if j < int(0.7 * k) else "val" if j < int(0.85 * k) else "test"

    rows, counts = [], {}
    for f in sorted(m):
        r = m[f]
        s = file_split.get(f) or split_of[r["speaker_id"]]
        rows.append({"file": f, "speaker_id": r["speaker_id"], "split": s})
        c = counts.setdefault(s, [0, 0])
        c[int(r["label"])] += 1
    write_csv(SPLITS_CSV, ["file", "speaker_id", "split"], rows)

    print(f"{n} speakers -> train {n_train} / val {n_val} / test {n_test} (seed {args.seed})")
    for s in ("train", "val", "test"):
        neg, posi = counts.get(s, [0, 0])
        print(f"  {s:<5} positives {posi:>5}   negatives {neg:>5}")
    print(f"Written to {SPLITS_CSV}")


def main():
    ap = argparse.ArgumentParser(description="Hey Wakey dataset recorder")
    ap.add_argument("--device", default=None, help="input device index or name")
    ap.add_argument("--list-devices", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    ss = sub.add_parser("speaker", help="record one speaker session")
    ss.add_argument("--pos-scale", type=float, default=1.0,
                    help="multiply the number of positive clips (1.0 = 70, 1.5 = about 105)")
    sub.add_parser("check", help="input level check")
    am = sub.add_parser("ambient", help="record background noise clips")
    am.add_argument("--label", required=True, help="e.g. tv_speech, fan, ac, street, kitchen, crowd, music")
    am.add_argument("--count", type=int, default=100)
    sp = sub.add_parser("splits", help="speaker independent 70/15/15 split")
    sp.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.device is not None and args.device.isdigit():
        args.device = int(args.device)
    if args.list_devices:
        print(get_sd().query_devices())
        return
    {"speaker": run_session, "check": run_check, "ambient": run_ambient,
     "splits": run_splits}.get(args.cmd, lambda a: ap.print_help())(args)


if __name__ == "__main__":
    main()
