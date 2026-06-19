# Bandit + mini-MID bonus (PsychoPy)

PsychoPy port of the jsPsych task. Three-arm probabilistic bandit with two
reversals and an interleaved adaptive-window mini-MID bonus block
(16 food + 14 neutral). Same data fields as the web export, plus a few
hardware/timing columns.

## Run

```
python bandit_mid_task.py
```

A startup dialog collects participant ID, session, an optional seed (blank draws
a random one, logged in every row), the food set (auto/sweet/savory), and the
iEEG options (photodiode square, trigger backend, serial port, parallel
address). Press SPACE at the instructions to begin. The experimenter can abort
at any time with Escape; data written so far is kept.

Requires PsychoPy (2023.2 or newer recommended). Install via the standalone
PsychoPy app or `pip install psychopy`. Serial triggers also need `pyserial`.

## Responses

- Bandit choice: the LEFT, DOWN, and RIGHT arrow keys select the left, middle,
  and right symbol. A 4 s nudge ("Please answer faster.") appears if no key is
  pressed, and the trial still waits for a response.
- Bonus target: press ANY key as fast as you can the moment the white square
  appears (PsychoPy keyboard, sub-ms RT). Pressing before the square is logged
  as "too soon".

## Stimuli

All stimuli live under `stimuli/`. Drop your images into these folders (any
.png/.jpg/.jpeg, auto-discovered):

```
stimuli/shapes/         square.png, circle.png, triangle.png (the bandit symbols)
stimuli/win/sweet/      sweet food images (win feedback + food bonus cue)
stimuli/win/savory/     savory food images (win feedback + food bonus cue)
stimuli/neutral/        neutral / scrambled images (neutral bonus cue)
stimuli/loss/           loss feedback images
```

Images are drawn with their aspect ratio preserved: each fits inside its display
box (the bonus cue inside a 0.5-of-height square, feedback and symbols inside
smaller boxes) without stretching, so non-square photos and scrambled images are
not distorted. Square 1024x1024 sources still work and simply fill the box.

If a bandit symbol file is missing, a plain dark square, circle, or triangle is
drawn instead, so the task runs before you add your own. Empty food/neutral/loss
folders fall back to a labelled box (cues) or a drawn sad face (losses).

## Task parameters

- Reward profiles (p_win/p_loss): 80/20 (EV +6), 50/50 (EV 0), 30/70 (EV -4).
- 200 bandit trials, two reversals at trials 69 and 130. Each reversal rotates
  all three profiles in a random direction (a 3-cycle, so every arm changes
  role, including the chance arm).
- Bandit timing: 400 ms choice animation, 900 ms feedback, then a jittered
  500-800 ms ISI (logged per trial as `isi_ms`).
- Bonus block: 16 food + 14 neutral, spread across the stream and buffered
  around both reversals. Sequence per bonus: "Bonus round!" (1 s), cue (1.5 s),
  anticipatory fixation jittered 1500-3000 ms, white square (adaptive window),
  500 ms grace, feedback (1.5 s).
- Adaptive window: starts 400 ms, hit shortens by 15 ms, miss lengthens by
  30 ms, floored at 250 ms and capped at 500 ms (converges near 66% hits).

## Points

The header shows task points and bonus points side by side throughout. Bonus
points are tracked on a separate tally and added to the task score for the
combined TOTAL on the end screen. Bonus rows never carry a bandit win/loss
outcome, so the bandit reward rate is unaffected.

## Output

One CSV per session in `data/`, written one row at a time and flushed (a crash
mid-session keeps everything up to the last completed trial). Rows are in
chronological order; `trial_type` is `bandit`, `bonus_food`, or `bonus_neutral`.
The column set matches the web export, plus `session`, `t_onset_s`, `isi_ms`,
and `trigger_code`. A matching `.log` records every event label with its marker
for offline alignment.

## Reproducibility

The bandit reward schedule uses the same mulberry32 generator and draw order as
the web version (symbol placement, slot order, then per-trial outcomes), so a
given seed reproduces the schedule. Note the lab spec differs from the web spec
in profiles and reversals (two reversals at 69 and 130, 80/20 best arm), so the
schedule matches the web version only where those settings match. Cosmetic draws
(food set, which pictures, corner tilt, ISI jitter, bonus deck and placement)
run on a separate seed-derived stream and are reproducible within PsychoPy.

## iEEG notes

- Triggers: every event sends the same marker, a comma. Choose `serial` (writes
  the byte `,` = 0x2C to the configured port) or `parallel` (writes the comma
  byte 44). With no device present the marker is logged only. Event identity is
  preserved in the `.log` labels and in the data file via `trial_type` and the
  onset-time columns. If your recording system needs a different transport or a
  distinct code per event, that is a small change in the `Triggers` class and
  `EVENT_CODES`.
- Photodiode: a white square pulses bottom-right at every event onset (cue,
  square, outcome). Reposition or resize `pd_stim` for your sensor.
- Timing is frame-locked; onset timestamps in the data are flip times.

## Decisions worth a look

- The web inactivity auto-abort was dropped (an experimenter is present); the
  per-trial 4 s nudge is kept.
- Loss feedback uses your `stimuli/loss/` images; the drawn sad face is only a
  fallback when that folder is empty.
