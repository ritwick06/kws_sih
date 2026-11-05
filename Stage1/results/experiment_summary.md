# Stage 1 Experiment Summary

Dataset: 285 one-second, 16-kHz mono WAV files:
- Positive: 70
- Hard negatives: 65
- Noise: 100
- General speech: 50

Main evaluation: 5-fold stratified cross-validation with references and thresholds selected within training folds.

| Method | TPR | TNR | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|---:|
| MFCC + single-template DTW | 78.57% | 87.91% | 83.24% | 85.61% |
| MFCC + 5-template DTW | 78.57% | 84.19% | 81.38% | 82.81% |
| MFCC + Delta + Delta-Delta + 5-template DTW | 84.29% | 87.44% | 85.86% | 86.67% |

The main method improved balanced accuracy by 2.62 percentage points over the single-template MFCC baseline.

Limitation: reliable speaker IDs were unavailable, so this is not speaker-independent evaluation.
