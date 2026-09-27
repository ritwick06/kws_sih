## Hybrid Edge-KWS + Cloud-ASR System

A Complete Solution Architecture for Ultra-Lightweight,

High-Accuracy Keyword Spotting on Resource-Constrained Edge Hardware

Smart India Hackathon — Problem Statement Response

Custom Wake Keyword: "Hey Wakey" | Target Hardware: ESP32-WROOM-32

| Constraint | Target | Design Response |
| --- | --- | --- |
| RAM (system-wide, including | < 256 KB | ~150–210 KB tuned peak (see Section 9) |
| networking) |   |   |
| CPU utilisation while idling | < 10% | < 5% analytically (Section 9.2) |
| Framework | Open-source / TinyML | TensorFlow Lite for Microcontrollers |
|   | only |   |
| Keyword | Custom, not a generic | "Hey Wakey", trained from scratch |
|   | assistant word |   |
| Evaluation hardware | ESP32 / Raspberry Pi | ESP32-WROOM-32, dual core, FreeRTOS |
|   | class MCU |   |

This document presents the complete engineering rationale, mathematics, dataset methodology, firmware architecture and evaluation plan for the proposed system, from raw microphone samples to a confirmed cloud

handoff.


## Table of Contents

- 1. Executive Summary

- 2. Problem Restatement and Design Philosophy

- 3. System Architecture Overview

- 4. Stage 0/1 — The Adaptive Listening Gate (Signal-Level Mathematics)

- 5. Stage 2 — Neural Keyword Verification

- 6. Dataset Engineering Methodology

- 7. Training and Quantization Pipeline

- 8. Hardware Realisation on ESP32-WROOM-32

- 9. Resource Budget Analysis (RAM, CPU, Latency)

- 10. Cloud Handoff and Streaming Mechanism

- 11. Optimisation for Strict Constraint Compliance

- 12. Evaluation Methodology

- 13. Problem Statement Compliance Matrix

- 14. Engineering Novelty and Justification

- 15. Risks, Limitations and Future Work

- 16. Conclusion


## 1. Executive Summary

Voice-first IoT interaction is bottlenecked today by a single bad assumption: that every spoken second must be sent to the cloud to be understood. This is expensive at the scale of millions of devices, invasive to user privacy because it implies a continuously streaming microphone, and slow because every interaction pays a network round trip even for a simple wake event. The industry answer, and the one this problem statement asks teams to build, is a hybrid split: a tiny always-on model on the device decides only one thing — was the wake keyword just spoken — and only on a positive decision does the device wake the network stack and forward audio to a full Automatic Speech Recognition (ASR) service in the cloud.

This document presents a complete, implementable answer to that problem for the custom keyword "Hey Wakey", targeting an ESP32-WROOM-32 microcontroller. The system is built as a two-stage detection cascade running on one CPU core, paired with a independently scheduled networking task on the second core, coordinated through FreeRTOS primitives. Stage 0/1 is a cheap, always-on digital signal processing (DSP) gate that filters obvious non-speech using energy, zero-crossing rate, and a small Goertzel-based spectral ratio. Stage 2 is a quantized, depthwise-separable convolutional neural network (DS-CNN) that makes the final, accuracy-critical decision, running only on the rare windows that survive Stage 1. Only a confirmed Stage 2 detection triggers the network core to leave modem-sleep and stream audio to the cloud ASR endpoint.

The design is built around one governing principle stated explicitly in the problem statement's evaluation rubric: efficiency, accuracy and latency are evaluated together, and the margin between them must stay small. Rather than treating this as three separate optimisation problems, this document treats it as one system-level constraint and shows, with worked mathematics, how each stage's design choices trade off against the others, and where the real risk in meeting the <256KB RAM / <10% CPU bar actually lies once the full networking and TLS stack is accounted for — not just the audio pipeline in isolation.

## 2. Problem Restatement and Design Philosophy

## 2.1 What is actually being asked

The problem statement does not ask for a speech recognition system. It asks for the gatekeeper that decides when the expensive speech recognition system in the cloud should be woken up, and the mechanism that gets the resulting audio there with minimal delay and minimal data overhead. This reframing matters for every design decision that follows: the on-device model's only job is a binary decision — keyword or not-keyword — repeated constantly, which is a dramatically smaller problem than full transcription, and is exactly why it can be solved in kilobytes of RAM rather than gigabytes.

## 2.2 The three metrics are coupled, not independent

Efficiency (footprint and idle CPU), accuracy (true-positive rate versus false-activation rate), and latency (keyword-end to cloud-received) are stated as three separate evaluation axes, but they are not independent in practice. A model tuned to never miss the keyword will also false-trigger more often, which directly increases average power draw and average cloud compute cost, because every false trigger wakes the network stack. A model shrunk aggressively to save RAM and CPU will typically lose accuracy. The correct way to reason about this system is therefore not "minimise footprint" or "maximise accuracy" in isolation, but to fix an explicit priority order and design every component against it.

## Stated priority order for this solution

Accuracy first, then compute, then power — with the explicit goal of keeping the margin between these three small, rather than sacrificing one heavily to gain slightly on another. This is operationalised throughout the document as: never degrade Stage 2's discriminative power to save a few kilobytes of RAM; instead, spend the RAM/CPU budget on the DSP gate and networking stack more carefully.

## 2.3 Why a single loudness threshold is explicitly rejected

An early design temptation is to gate purely on audio loudness (short-time energy). This is rejected in this architecture for a concrete reason: energy alone cannot distinguish a spoken word from a loud non-speech event (a door slam, a


dropped object, television audio) or distinguish a quiet, distant utterance of the keyword from ambient background noise. A pure loudness gate therefore either misses quiet keyword attempts (hurting true-positive rate) or false-triggers on any loud sound (hurting false-activation rate) — violating the accuracy-first priority. Production wake-word systems from major voice assistant vendors instead run a genuine lightweight classifier continuously on dedicated low-power silicon, followed by a second, higher-confidence pass before the full pipeline wakes. This architecture reproduces that same systems-level pattern using only open-source TinyML tooling on general-purpose microcontroller hardware, which is precisely what the problem statement's open-source and no-proprietary-SDK restrictions push teams toward.

## 3. System Architecture Overview

The system is organised as a five-stage pipeline distributed across the two CPU cores of the ESP32-WROOM-32, coordinated by FreeRTOS. The table below is the canonical reference for the entire document; every subsequent section expands one row of this table in full mathematical and implementation detail.

| Stage | Core | Function | Runs when | Approx. cost |
| --- | --- | --- | --- | --- |
| S0 — Capture | Core 1 | I2S DMA fills a circular audio buffer directly | Continuously, | ~0 CPU cycles |
|   |   | from the microphone | hardware-driven |   |
| S1 — DSP Gate Core 1 |   | Short-time energy + zero-crossing rate + | Every ~20–30 ms | <5% CPU |
|   |   | Goertzel speech-band ratio decide if the | window |   |
|   |   | window is worth examining further |   |   |
| S2 — NN Verify | Core 1 | Quantized DS-CNN on MFCC features | Only when S1 passes | 2–5M MACs per call, |
|   |   | gives the final keyword / not-keyword | (rare) | amortised <1% CPU |
|   |   | decision |   |   |
| S3 — Handoff | Core 1 | FreeRTOS task notification signals Core 0 | Only on confirmed | single-digit |
|   | Core 0 | the instant S2 confirms a detection | detection |   |
| S4 — Stream | Core 0 | WiFi leaves modem-sleep, opens (or | Only during an active | Active WiFi/TLS |
|   |   | reuses) a TLS session, streams buffered + | detection event | power, brief duration |
|   |   | live audio to the cloud ASR endpoint |   |   |

This structure is deliberately a cascade with monotonically increasing cost per stage: the cheapest possible operation (hardware DMA) runs permanently, the next-cheapest (DSP gate) runs continuously but on tiny features, the expensive operation (neural network) runs rarely, and the most expensive operation (radio transmission) runs only on a confirmed positive. This is the structural reason the system can simultaneously target high accuracy and near-zero idle cost — the expensive, accurate component is not avoided, it is simply invoked conditionally.

## 4. Stage 0/1 — The Adaptive Listening Gate

Stage 1 must satisfy a specific and slightly counter-intuitive design goal: it should be biased toward high recall, not high precision. Its job is not to make the final keyword decision — that is Stage 2's job — it is to cheaply discard windows that are obviously not speech, while letting through anything that plausibly could be. Every operation in this stage is chosen to cost only a handful of integer multiply-adds per sample, because it is the only component of the pipeline that runs on every single audio window, forever.

## 4.1 Short-Time Energy (STE)

For a window of N audio samples x[0..N-1], the normalised short-time energy is:

Worked example, using the sample window [2, -2, 1, 3, -4] with N = 5:

Energy is normalised by window length so the decision threshold does not depend on the chosen window size, and the ambient noise floor is re-estimated every few seconds (a running minimum or a slow exponential average of E during


quiet periods) so the same absolute threshold does not fail differently in a quiet room versus a noisy room.

## 4.2 Zero-Crossing Rate (ZCR)

Energy alone cannot separate a loud non-speech event from real speech. Zero-crossing rate, the rate at which the signal changes sign, is a second near-free feature that helps here: voiced speech occupies a characteristic mid-range ZCR band, while pure tones or hums sit low and broadband noise (claps, static) sits high.

```
ZCR = (1 / (2N)) * sum( | sign(x[n]) - sign(x[n-1]) | ) for n = 1 .. N-1
```

This requires only sign comparisons — no multiplication at all — making it essentially free to compute alongside the energy calculation on the same window.

## 4.3 The Goertzel Algorithm — a spectral fingerprint without an FFT

A full Fast Fourier Transform is unnecessary and comparatively expensive when only a handful of frequency bins in the speech band are of interest. The Goertzel algorithm computes the power at a single target frequency bin k (out of an N-point transform) using a simple second-order recurrence, at a cost of one multiply-add per input sample per bin of interest.

```
coeff = 2 * cos(2*pi*k / N)
for each sample x[n]:
s0 = x[n] + coeff * s1 - s2
s2 = s1
s1 = s0
power_k = s1^2 + s2^2 - coeff * s1 * s2
```

Applied to 4–6 bins spanning roughly 300 Hz–3.4 kHz (the classical telephony speech band, e.g. centred near 500 Hz, 1000 Hz, 1500 Hz, 2000 Hz, 2500 Hz, 3000 Hz at a 16 kHz sample rate with a 256-sample window), this produces a coarse but genuinely spectral fingerprint at a cost of roughly 5 × 256 = 1280 multiply-adds per window — still orders of magnitude cheaper than a 256-point FFT, which requires on the order of N log(N) 2048 complex operations, each substantially more expensive than a single Goertzel multiply-add.

## 4.4 The Stage 1 decision variable

The final gating decision combines all three cheap features into a single speech-presence ratio:

```
R = ( sum of Goertzel power across speech-band bins ) / E_total
```

The gate opens (Stage 2 is invoked) when R exceeds a threshold tuned on a validation set, combined with a ZCR band check to reject broadband noise and pure tones. Because R is a ratio of speech-band energy to total energy rather than raw energy, it is far more robust to overall loudness than a plain energy threshold: a loud non-speech event has low R (its energy is not concentrated in the speech-band bins), while even a moderately quiet spoken word has comparatively high R.

## Design discipline for this stage

The threshold on R is deliberately tuned loose. False positives that reach Stage 2 are cheap — they cost one additional neural network inference, which is itself inexpensive when amortised. False negatives at this stage are expensive in a way that cannot be recovered later — if Stage 1 discards a genuine utterance, Stage 2 never gets the chance to correct it. This is why the system-level true-positive rate is the product of each stage's recall (see Section 12.2), which mathematically forces Stage 1 to be recall-biased if the overall system is to hit a high true-positive target.

## 5. Stage 2 — Neural Keyword Verification

## 5.1 Feature extraction: from raw audio to MFCCs

The audio windows that survive Stage 1 are converted from raw waveform into Mel-Frequency Cepstral Coefficients (MFCCs) — the standard, well-understood input representation for small-footprint keyword spotting. This conversion is what allows a tiny model to generalise across speakers and recording conditions instead of memorising raw waveform


shapes.

## Step 1 — Framing

The buffered 1-second audio window (16 kHz, mono) is split into overlapping frames of 25 ms (400 samples) with a 10 ms hop (160 samples):

```
num_frames = floor( (16000 - 400) / 160 ) + 1 = 98 frames (commonly pooled/strided down to ~49)
```

## Step 2 — Windowing

Each frame is multiplied by a Hamming window to reduce spectral leakage before the transform:

```
w[n] = 0.54 - 0.46 * cos( 2*pi*n / (N - 1) ) for n = 0 .. N-1
```

## Step 3 — Power spectrum

A 512-point FFT is applied to each windowed frame, and the power spectrum is retained for the first 257 bins (0 to Nyquist):

```
P[k] = |X[k]|^2 for k = 0 .. 256
```

## Step 4 — Mel filterbank

Frequencies are mapped onto the perceptually-motivated Mel scale:

```
m = 2595 * log10( 1 + f / 700 )
```

Approximately 26 triangular filters, evenly spaced in Mel-space, are applied to the power spectrum, each filter summing the power spectrum energy that falls under its triangular response — producing 26 filterbank energies per frame.

## Step 5 — Log and Discrete Cosine Transform

The log of each filterbank energy is taken (compressing dynamic range in a way that matches human loudness perception), followed by a Discrete Cosine Transform that decorrelates the filterbank outputs and compacts the useful information into the lowest coefficients:

```
MFCC[i] = sum_j( log(FB[j]) * cos( pi * i * (j + 0.5) / 26 ) ) keep i = 0 .. 12
```

The first 13 coefficients are retained. For a 1-second buffered clip this produces a feature matrix of approximately 49 × 13 values — the input tensor to the neural network.

## 5.2 Why depthwise-separable convolution

A standard 2D convolution over a Dk × Dk kernel, with M input channels, N output channels, applied over a Df × Df feature map, costs:

```
Cost_standard = D_k^2 * M * N * D_f^2 multiply-adds
```

A depthwise-separable convolution factorises this into a per-channel spatial pass (depthwise) followed by a 1×1 channel-mixing pass (pointwise):

```
Cost_separable = D_k^2 * M * D_f^2 + M * N * D_f^2 multiply-adds
```

The ratio of separable cost to standard cost is:

```
Cost_separable / Cost_standard = 1/N + 1/D_k^2
```

For a typical configuration in this system — a 3×3 kernel and 64 output channels — this ratio is approximately 1/64 + 1/9 0.126, an 8–9× reduction in multiply-add operations for near-identical representational capacity. This is the single largest lever available for keeping Stage 2 cheap without sacrificing the accuracy that a convolutional architecture provides over a plain fully-connected network.

## 5.3 Reference architecture

Input:

Conv2D:

DS-Conv block x3: depthwise 3x3 + pointwise 1x1, 64 channels -> BatchNorm -> ReLU

GlobalAveragePool

Fully Connected: 64 -> 2 (softmax: {keyword, not-keyword})

49 x 13 x 1 (MFCC feature matrix)

10 filters, 4x10 kernel, stride (2,1) -> BatchNorm -> ReLU


This architecture totals approximately 15,000–25,000 trainable parameters. Batch normalisation layers are folded into the preceding convolution's weights and bias during export — there is no separate batch-normalisation computation at inference time; it is algebraically absorbed into the same multiply-accumulate pass, at zero additional runtime cost.

## 5.4 Why DS-CNN over the alternatives

| Architecture | Relative accuracy | Relative footprint | Verdict |
| --- | --- | --- | --- |
| Plain DNN (fully-connected on | Lowest (~85-90%) | Smallest | Too weak for the accuracy-first |
| flattened MFCCs) |   |   | priority |
| DS-CNN (chosen) | High (94%+ achievable) Small (~20-25KB int8) |   | Best accuracy-per-compute; |
|   |   |   | chosen |
| CRNN (conv + GRU) | Marginally higher | Larger, more RAM for | Diminishing return for this budget |
|   |   | recurrent state |   |

Published small-footprint keyword spotting benchmarks on microcontroller-class hardware consistently find depthwise-separable CNN architectures sitting at the most favourable point on the accuracy-versus-compute curve for this problem size, which is the basis for selecting it over a plain DNN or a recurrent architecture here.

## 6. Dataset Engineering Methodology

## 6.1 Positive class — "Hey Wakey"

- Recorded from as many distinct speakers as the team can source — minimum target of several hundred clean utterances, more is strictly better for generalisation.

- Varied recording distance from the microphone (near-field and far-field, e.g. 0.3 m to 3 m).

- Varied speaking pace, stress pattern, and pitch across recordings of the same speaker where possible.

- Varied microphone angle relative to the mouth, since a real deployment will not guarantee a fixed geometry.

## 6.2 Negative class

Negatives must substantially outnumber positives, which is normal for keyword spotting and is corrected for during training with class weighting rather than by discarding negative data. Two sources are used:

- 1 Public corpus negatives — other spoken words, silence, and categorised background noise clips drawn from an open dataset such as Google Speech Commands, providing broad phonetic coverage the team could not feasibly record themselves.

- 2 Environment-specific negatives — the team's own recordings of television audio, household background noise, and phrases that sound superficially similar to "Hey Wakey", since these are the false-activation triggers most likely to actually occur in the demo environment.

## 6.3 Normalisation

- Resample every clip to 16 kHz mono.

- Trim or zero-pad every clip to exactly 1.0 second.

- Peak-normalise amplitude so recording-level gain differences do not become a spurious feature the model latches onto.

## 6.4 Augmentation — why it matters more than raw dataset size

Augmentation is what separates a model that performs well in a quiet demo room from one that performs well in the noisy, unpredictable conditions of a live evaluation. Three transformations are applied stochastically during training:

| Augmentation | Parameter range | What it protects against |
| --- | --- | --- |
| Time shift | ±100 ms | Imperfect alignment of the keyword within the 1s window |
| Background noise mixing | 0–20 dB SNR | Real ambient noise at evaluation time |


| Pitch shift | ±10% | Speaker-to-speaker pitch variation |
| --- | --- | --- |
| Speed perturbation | 0.9–1.1× | Speaking-rate variation |

## 6.5 Speaker-independent splitting

The train/validation/test split (a reasonable starting ratio being 70/15/15) is performed by speaker, not by clip: no speaker whose voice appears in the training set may also appear in validation or test. This is a strict requirement, not a nicety — splitting by clip instead of by speaker allows the model to partially memorise speaker-specific characteristics, producing a validation accuracy number that looks excellent but does not predict real-world performance on an unseen speaker at the actual demo.

## 6.6 Feature extraction and storage

The MFCC pipeline described in Section 5.1 is applied to every clip in the processed dataset, and the resulting 49×13 feature matrices (not the raw audio) are what is actually stored and fed to the training loop, typically serialised as .npy arrays or TFRecord shards with an accompanying label.

## 7. Training and Quantization Pipeline

## 7.1 Where training happens

All training happens once, offline, entirely off the ESP32, on a development machine using TensorFlow/Keras. The device never trains, never computes a gradient, and never updates a weight. This is a deliberate and important architectural separation: the expensive, iterative optimisation process runs where compute is abundant, and the frozen output of that process is what gets deployed to the resource-constrained edge target.

## 7.2 Training procedure

- 1 Build the DS-CNN architecture from Section 5.3 in Keras.

- 2 Train with class-weighted categorical cross-entropy, since the negative class substantially outnumbers the positive class in the raw dataset.

- 3 Track validation precision and recall separately, not just accuracy — accuracy alone can look deceptively good on an imbalanced dataset while hiding a poor false-activation rate.

- 4 After training converges, sweep the softmax decision threshold against the validation set and plot the resulting precision/recall (or ROC-style) curve, then select the operating threshold deliberately rather than defaulting to 0.5.

## 7.3 Quantization

Post-training int8 quantization is applied using the TensorFlow Lite converter with a representative dataset (a few hundred real MFCC samples drawn from the training distribution) so the converter can correctly calibrate the dynamic range of each activation tensor. Int8 weights are approximately four times smaller than the equivalent float32 weights, which is the direct source of the ~20–25 KB model footprint figure used throughout this document, and int8 arithmetic is what allows the model to run efficiently on a microcontroller core with no dedicated floating-point acceleration for this class of workload.

Quantization is not free of cost — it introduces a small accuracy degradation relative to the full-precision model. Careful calibration keeps this degradation small (published low-bit-width KWS accelerator work reports well under one percentage point of accuracy loss relative to full precision when quantization is done properly), and this degradation must be measured directly on the held-out test set for this project, not assumed.

## 7.4 Export

The quantized model is converted to a C byte array for firmware embedding:

xxd -i model.tflite > model_data.cc

This byte array becomes a read-only data section compiled directly into the firmware image, requiring no filesystem or dynamic loading at runtime.


## 8. Hardware Realisation on ESP32-WROOM-32

## 8.1 Why FreeRTOS, and why it needs no additional justification

The ESP32 boots FreeRTOS by default under the ESP-IDF toolchain — it is not an addition to the system, it is the operating substrate the entire firmware is written against. The relevant engineering work is not "whether to use FreeRTOS" but how to use its primitives — explicit core pinning, event-driven blocking instead of polling, and lightweight cross-core signalling — to achieve the near-zero-idle-cost property the problem statement requires.

## 8.2 Core assignment

The ESP-IDF WiFi and Bluetooth protocol stack is pinned to Core 0 by default. This is a fixed property of the platform, not a configuration choice, and it directly determines the correct task placement for this system:

| Core | Hosts | Rationale |
| --- | --- | --- |
| Core 1 (APP_CPU) | Audio capture, Stage 1 DSP gate, Stage | Isolated from WiFi stack interrupts and scheduling jitter, |
|   | 2 neural inference | which protects the latency-critical realtime audio path |
| Core 0 (PRO_CPU) | WiFi/TCP-IP stack, TLS session, | Co-located with the networking stack that already runs here; |
|   | streaming task | avoids duplicating network-stack overhead across cores |

## 8.3 Task and signalling design

- Audio capture uses the I2S peripheral's DMA engine to fill a circular buffer directly from the microphone with zero CPU cycles spent on sample-by-sample transfer — the hardware moves the bytes.

- The Core 1 audio task blocks on the I2S driver's event queue (xQueueReceive) waiting for a "buffer ready" notification, rather than polling in a busy loop. This is what allows the core to actually enter a low-power idle state between DMA events instead of spinning at full clock.

- Stage 1 and Stage 2 run inside this same Core 1 task, invoked only when their respective preceding stage's condition is met.

- Feature extraction (the FFT and Goertzel computations of Sections 4 and 5) uses esp-dsp, Espressif's Xtensa-optimised DSP library, rather than a hand-rolled implementation, since the optimised kernels materially affect how much of the CPU budget Stage 1/2 actually consume.

- Neural network inference uses esp-tflite-micro, the official Espressif port of TensorFlow Lite for Microcontrollers, which in turn uses esp-nn — Xtensa-accelerated int8 kernels — giving faster inference than the generic reference TFLite Micro kernels.

- On a confirmed Stage 2 detection, the Core 1 task sends a single-bit FreeRTOS task notification (xTaskNotifyFromISR / xTaskNotify) to the Core 0 networking task. A task notification is used in preference to a full queue because the payload is a single flag, not data — this is the cheapest possible cross-core signalling primitive FreeRTOS offers for this exact case.

## 8.4 Power management

- esp_pm_configure with automatic light-sleep and dynamic frequency scaling enabled, so the CPU clock and sleep state track actual workload rather than running at a fixed maximum clock permanently.

- Tickless idle enabled in the FreeRTOS configuration, allowing the scheduler itself to cooperate with light-sleep entry between events rather than waking on every RTOS tick regardless of whether there is work to do.

- WiFi held in modem-sleep (esp_wifi_set_ps(WIFI_PS_MIN_MODEM)) rather than fully disconnected while idle — this is a deliberate trade of a small amount of extra idle power against avoiding the hundreds-of-milliseconds-to-seconds reconnect delay a full disconnect would otherwise impose the moment a keyword is confirmed, which would directly and severely harm the latency metric.

## 9. Resource Budget Analysis


## 9.1 RAM budget — idle state

| Component | Estimated footprint |
| --- | --- |
| Audio ring buffer (I2S DMA, ~0.5-1s @ 16kHz/16-bit mono) | 16-32 KB |
| MFCC feature buffer (49x13, fixed-point) | 1-3 KB |
| DS-CNN weights (int8 quantized) | 20-25 KB |
| TFLite Micro interpreter arena (activations/scratch) | 10-20 KB |
| FreeRTOS task stacks (2 tasks) | 4-8 KB |
| Goertzel state and misc DSP buffers | ~5 KB |
| WiFi driver static buffers (resident even in modem-sleep) | ~32-35 KB |
| lwIP TCP/IP stack baseline | 20-40 KB |
| Application-level idle total (approx.) | ~110-170 KB |

This leaves comfortable headroom under the 256 KB ceiling while idling — the state the CPU utilisation constraint explicitly refers to.

## 9.2 CPU budget — idle state (Stage 1 only)

The problem statement's 10% CPU figure is explicitly scoped to "idling in continuous listening mode", which corresponds precisely to Stage 1 running continuously; Stage 2 and the streaming task are, by definition, not part of the idle state.

```
Goertzel (5 bins) + energy + ZCR per ~256-sample window: ~3,000-4,000 operations
Window cadence: ~20-30 ms -> ~33-50 windows/sec
Stage 1 operation rate: ~130,000-200,000 ops/sec
ESP32 core @ 240 MHz, 10% CPU budget: ~24,000,000 cycles/sec
Pessimistic 4-5 cycles per operation (incl. RTOS overhead): ~600,000-1,000,000 cycles/sec
```

Result: Stage 1 alone consumes well under 5% of the 10% budget.

## 9.3 The peak-load risk: RAM during an active streaming event

The RAM ceiling in the problem statement, unlike the CPU figure, is not scoped to the idle state only — the constraint explicitly covers the handoff and cloud-streaming path as well. This is the single largest compliance risk in the system, and it must be engineered for directly rather than assumed away, because a live TLS session is genuinely heavy on a microcontroller.

| Component (active during streaming only) | Typical footprint |
| --- | --- |
| mbedTLS per-connection SSL context (mbedtls_ssl_setup) | ~23 KB |
| TLS IN/OUT record buffers, default configuration | 16 KB each direction = 32 KB |
| Transient handshake allocation (certificates, keys, temp buffers) | ~40-50 KB, freed after handshake |

Summed naively against the idle baseline in Section 9.1, this can exceed the 256 KB ceiling during an active streaming event under default ESP-IDF/mbedTLS settings. Section 11 details the specific, documented configuration changes that bring this back under budget.

## 9.4 Latency budget

The metric being measured is explicitly the delta between the keyword ending and the cloud ASR server receiving the audio stream — a systems/pipeline metric, not a model-inference-speed metric in isolation. The contributing components are:

| Component | Contributes to latency because |
| --- | --- |
| Stage 2 inference time | Detection confirmation must complete before handoff can begin |


| FreeRTOS task notification (Core 1 to Core | Cross-core signalling delay, single-digit microseconds |
| --- | --- |
| 0) |   |
| WiFi wake from modem-sleep | Radio must leave low-power state before transmission is possible |
| TLS session establishment or reuse | A fresh handshake costs far more than reusing a kept-alive session |
| Network transmission of the buffered audio | Physical transmission time over the WiFi link |

Keeping WiFi in modem-sleep rather than fully disconnected (Section 8.4) is the single largest lever against this budget, since it removes the WiFi association/authentication delay from the critical path entirely, leaving only the much smaller modem wake time.

## 10. Cloud Handoff and Streaming Mechanism

## 10.1 Sequence of events

- 1 Stage 2 confirms a keyword detection on Core 1.

- 2 Core 1 sends a FreeRTOS task notification to the Core 0 streaming task — a single flag, no payload copy.

- 3 Core 0's streaming task wakes, instructs the WiFi driver to leave modem-sleep.

- 4 A TLS session is established or, where session resumption is supported by the cloud endpoint, an existing session is reused to avoid a full handshake.

- 5 The already-buffered audio (captured continuously by the Stage 0 ring buffer, so the beginning of the utterance is never lost even though detection completes slightly after the utterance itself) plus the live continuing audio is streamed to the cloud ASR endpoint.

- 6 Streaming continues until an end-of-utterance condition is met — either a short silence/VAD timeout or a fixed maximum duration, whichever the team selects and states explicitly in the design.

- 7 The streaming task closes or parks the session, WiFi returns to modem-sleep, and Core 0 returns to its blocked, near-zero-cost idle state awaiting the next notification.

## 10.2 Why the raw audio buffer decouples capture from detection latency

Because Stage 0's DMA-driven ring buffer captures continuously and independently of when Stage 1/2 actually get around to examining a given window, the system never loses the beginning of an utterance merely because detection completed a little late. The buffered window handed to the streaming task always contains the full utterance context, not just the audio captured after the detection instant.

## 11. Optimisation for Strict Constraint Compliance

The following configuration changes, all documented, standard ESP-IDF/mbedTLS options, are what bring the peak RAM figure from Section 9.3 back under the 256 KB ceiling, without weakening the security of the cloud connection.

| Lever | Mechanism | Effect |
| --- | --- | --- |
| Asymmetric TLS content length Reduce the TLS OUT buffer (the direction carrying |   | Frees up to ~14-24 KB of dynamic |
|   | outgoing audio, fully under this project's control) from the | memory per session |
|   | 16 KB default down to as little as 2-4 KB |   |
| CONFIG_MBEDTLS_DYNAMI | mbedTLS frees its dynamic allocations (buffers, | Reduces peak heap usage |
| C_BUFFER | certificates, keys) immediately after each use rather than | significantly, at a small performance |
|   | holding them for the session's full lifetime | cost from more frequent allocation/free |
|   |   | cycles |
| Reduced WiFi/lwIP static buffer | Lower the number of static WiFi buffers and cap | Trades some network throughput |
| counts | dynamic buffer counts via ESP-IDF Kconfig options | headroom (not needed for a |
|   |   | low-bitrate audio stream) for lower |
|   |   | steady-state memory |


Drop peer certificate post-handshake

Disable retention of the peer certificate once the handshake completes, where the application does not need it afterward

Further reduces per-session memory footprint

Applying these levers is expected to bring peak system RAM during an active streaming event down to roughly 150–210 KB, comfortably inside the 256 KB ceiling with the resulting configuration — though this estimate must be confirmed by direct on-device measurement (Section 12), not accepted on the strength of the arithmetic alone.

## 11.1 Open question worth raising with organisers

Whether the 256 KB figure is intended to cover the application layer alone or the entire running system including the WiFi/TLS stack materially changes the effective budget. This is directly analogous to the ESP32-versus-Raspberry-Pi hardware ambiguity already present in the problem statement's example hardware list, and is worth raising as an explicit clarifying question during the ideation round rather than assumed silently in either direction.

## 12. Evaluation Methodology

A design is only as credible as the evidence behind it. Each metric in the problem statement's rubric is paired here with the specific, concrete measurement procedure that will be used to report it — the deliverable is measured numbers on physical hardware, not claimed ones.

## 12.1 Efficiency — footprint and idle CPU

- RAM/Flash footprint: read directly from the ESP-IDF build's linker map file (.map), which reports exact static and dynamic allocation sizes per component, cross-checked at runtime with heap_caps_get_free_size() sampled before and during an active streaming event to capture the real peak, not just the theoretical one.

- Idle CPU utilisation: measured using FreeRTOS's built-in runtime statistics (vTaskGetRunTimeStats) under two explicit conditions — a quiet room (Stage 1 rarely opens the gate) and a noisy room (Stage 1 opens frequently) — since the second condition, not the theoretical DSP-only estimate, is the number that will actually withstand scrutiny if challenged.

## 12.2 Accuracy — true-positive rate and false-activation rate

For a two-stage cascade, the system-level rates are the product of each stage's individual rate, which is the precise mathematical reason Stage 1 must be recall-biased:

```
TP_system = TP_stage1 x TP_stage2
FA_system = FA_stage1 x FA_stage2
```

Worked illustration: if Stage 1 passes through 10% of non-keyword sounds and Stage 2 correctly rejects 95% of what reaches it, the system-level false-activation rate is 10% × 5% = 0.5%. The same multiplication applies to true positives — if Stage 1 misses 8% of genuine utterances and Stage 2 misses a further 5% of what reaches it, the system-level true-positive rate is only 92% × 95% 87%, not 95%. This is why Stage 1's threshold must be tuned for near-100% recall on the validation set: any recall lost at Stage 1 is recall the system can never recover, regardless of how accurate Stage 2 is.

The reported metric is a full confusion matrix (true positive, false positive, true negative, false negative counts) on the speaker-independent held-out test set from Section 6.5, together with the precision/recall curve used to select the Stage 2 decision threshold, not a single accuracy percentage.

## 12.3 Latency — keyword-end to cloud-received

Instrumented directly with timestamps: t0 is logged at the instant Stage 2 confirms detection (which corresponds to the keyword having ended, since Stage 2 operates on the buffered window after the utterance), and t1 is logged by the cloud endpoint at the instant it receives the first byte of the streamed audio. The reported latency is t1 minus t0, averaged across a representative number of live detection trials, not a single best-case measurement.

## 13. Problem Statement Compliance Matrix


| Requirement | Design response | Status |
| --- | --- | --- |
| Runs locally, streams to remote | DS-CNN local match on Core 1; FreeRTOS notification triggers Core 0 | Met |
| ASR on detection | stream |   |
| Open-source only, no | TensorFlow Lite Micro, esp-tflite-micro, esp-dsp — entirely open tooling | Met |
| proprietary SDK |   |   |
| Allowed TinyML framework | TensorFlow Lite for Microcontrollers, explicitly named as recommended | Met |
| No pre-trained generic keyword Custom-trained end to end on "Hey Wakey" from a purpose-collected dataset Met |   |   |
| Not a heavy/uncompressed | DS-CNN, int8 quantized, ~20-25KB — the explicit opposite of what is | Met |
| transformer | disqualified |   |
| CPU <10% while idling | Stage 1 DSP gate only, analytically well under 5% | Met, pending |
|   |   | device |
|   |   | measurement |
| RAM <256KB (incl. | ~110-170KB idle; ~150-210KB peak after mbedTLS/lwIP tuning | Achievable with |
| handoff/streaming) |   | tuning, requires |
|   |   | on-device |
|   |   | confirmation |
| High true-positive, near-zero | Cascade design with Stage 1 recall-biased, Stage 2 precision-driven | Architecturally |
| false activation |   | supported; |
|   |   | requires |
|   |   | trained-model |
|   |   | measurement |
| Evaluated on physical | All figures in this document are estimates pending on-device validation per | Planned |
| ESP32/RPi hardware | Section 12 |   |

## 14. Engineering Novelty and Justification

It is important to state plainly what is, and is not, novel about this solution, since overstating algorithmic novelty is a weak position against informed evaluators. Individually, Goertzel-based voice activity detection, depthwise-separable CNN keyword classifiers, int8 quantization, and multi-stage confidence cascades are all established techniques, and comparable cascade patterns are used in production wake-word systems from major voice assistant vendors.

## What is genuinely being contributed here

- 1 Reproducing a dedicated-silicon design pattern on commodity hardware. Production systems typically achieve this cascade using purpose-built low-power neural decision processor chips. This solution deliberately reconstructs the same systems-level pattern — cheap always-on filter, confirmatory neural network, gated cloud handoff — using only an open-source software stack on a general-purpose microcontroller, which is precisely the constraint the problem statement's open-source and no-proprietary-SDK restrictions are steering every team toward.

- 2 Core assignment matched to ESP-IDF's actual runtime behaviour. Rather than a generic "split work across two cores" approach, task placement is derived from the documented fact that the WiFi stack is pinned to Core 0 by default, and the realtime audio path is isolated from it accordingly.

- 3 Treating accuracy, compute and power as one coupled constraint with an explicit priority order, rather than optimising each metric independently, including the mathematical demonstration (Section 12.2) of exactly how cascade design choices propagate into system-level accuracy.

- 4 Transparent accounting of the full networking/TLS memory cost, including the peak-load risk during an active streaming event that a footprint analysis limited to the audio/DSP/NN path alone would miss entirely, together with the specific, documented ESP-IDF configuration levers used to bring it back under budget.

The correct framing for this submission is therefore disciplined systems engineering under real, fully-accounted-for constraints, rather than a claim of new machine learning technique. This is both a more accurate description of the work and, for a hardware-constrained hackathon evaluated on physical silicon, arguably the harder and more valuable thing to get right.


## 15. Risks, Limitations and Future Work

## 15.1 Known risks

- RAM during peak streaming is the tightest constraint in the system and depends on mbedTLS/lwIP tuning being implemented correctly, not just planned; this must be measured on real hardware before being claimed in the final presentation.

- Stage 1 threshold sensitivity: if tuned too tight, system-level true-positive rate suffers irrecoverably (Section 12.2); if tuned too loose, Stage 2's average trigger rate rises, which could push idle CPU utilisation measured in a genuinely noisy room closer to the 10% ceiling than the quiet-room analytical estimate suggests.

- Dataset size and diversity directly bound achievable accuracy; the figures cited in this document from published KWS benchmarks are representative targets, not guarantees, for this specific custom keyword and this specific team's collected data.

- Quantization-induced accuracy loss must be measured directly by comparing the full-precision and quantized model on the same held-out test set, rather than assumed from general published figures.

## 15.2 Future work beyond the hackathon scope

- On-device incremental adaptation to a specific user's voice, subject to the fixed-point training difficulties this introduces on microcontroller-class hardware.

- Multi-microphone far-field enhancement (beamforming, noise cancellation) for improved accuracy at greater distances than a single-microphone system can reliably support.

- Extension to a small vocabulary of on-device command words handled entirely locally, reserving the cloud ASR handoff only for open-ended queries.

## 16. Conclusion

This document has presented a complete, mathematically grounded, and implementation-ready architecture for the hybrid edge-KWS + cloud-ASR problem statement, built around a custom keyword, "Hey Wakey", and targeting the ESP32-WROOM-32. Every component — the DSP gate, the quantized neural verifier, the dual-core FreeRTOS task structure, the cloud handoff mechanism, and the resource budget — has been derived from first principles and connected back explicitly to the problem statement's stated evaluation metrics and technical boundaries.

The design is deliberately honest about what remains to be proven rather than merely designed: the peak RAM figure under active streaming, the true-positive and false-activation rates of the trained model, and the idle CPU utilisation in a realistically noisy environment all require on-device measurement before they can be claimed with confidence. The evaluation methodology in Section 12 exists specifically to close that gap between architecture and evidence, which is the work remaining between this document and a finished, demonstrable submission.

End of document.
