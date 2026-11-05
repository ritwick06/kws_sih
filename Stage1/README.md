# Hey Wakey — Stage 1 Pre-Trigger

Objective: develop a lightweight pre-trigger for the custom wake word **Hey Wakey** before activating a heavier AI model.

Pipeline:
**Microphone -> MFCC + Delta + Delta-Delta -> 5-reference DTW -> minimum DTW score -> threshold -> Trigger/Reject -> AI**

### 5-fold stratified CV result
- TPR: **84.29%**
- TNR: **87.44%**
- Balanced Accuracy: **85.86%**
- Accuracy: **86.67%**

### Repository
- `code/` — reference implementation
- `results/` — metrics and experiment summary
- `graphs/` — experiment plots
- `research/` — research notes

The 285 audio files used for the experiment are not included because they are large.

**Limitation:** reliable speaker IDs were not available, so this result must not be described as speaker-independent evaluation.

**Next:** collect genuine multi-speaker recordings and use speaker-grouped validation.
