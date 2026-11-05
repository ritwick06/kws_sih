# Stage 1 Research Notes

## Problem
A continuously running detector should reject unrelated speech before activating a heavier AI model.

## Proposed solution
Mic -> MFCC + Delta + Delta-Delta -> multi-reference DTW -> threshold -> AI trigger.

DTW aligns temporal speech patterns, allowing comparison when the same keyword is spoken at different rates.

Multiple references represent more than one valid acoustic pattern.

Delta and delta-delta features add temporal information to MFCC spectral features.

## Next experiment
Collect genuine recordings from multiple speakers and use GroupKFold or leave-one-speaker-out validation. Select references and thresholds only from training speakers and evaluate on unseen speakers.
