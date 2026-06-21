# Bandit + mini-MID bonus (PsychoPy, v13)

PsychoPy port of the jsPsych task. Three-arm probabilistic bandit with two
reversals and an interleaved adaptive-window mini-MID bonus block
(16 food + 14 neutral, phase-stratified). Same core data fields as the web
export, plus a few hardware, timing, and provenance columns.

## Run

```
python bandit_mid_task_v13.py
```

A startup dialog collects participant ID, session, an optional seed (blank draws
a random one, logged in every row), the food set (auto/sweet/savory/sweet+savory),
and the iEEG options (photodiode square, trigger backend, serial port, parallel
address). Press SPACE at the instructions to begin. The experimenter can abort at
any time with Escape; data written so far is kept.

Reruns never overwrite: each run gets its own numbered output folder (see
Output below), so the same ID and session can be entered repeatedly.

Requires PsychoPy (2023.2 or newer recommended). Install via the standalone
PsychoPy app or `pip install psychopy`. Serial triggers also need `pyserial`.

## Responses

- Bandit choice: the LEFT, DOWN, and RIGHT arrow keys select the left, middle,
  and right symbol. An arrow shape under each slot shows the mapping (the arrows
  are drawn as vector shapes, not font glyphs, so they render consistently). A
  4 s nudge ("Please answer faster.") appears if no key is pressed, and the trial
  still waits for a response.
- Bonus target: press ANY key as fast as you can the moment the square appears
  (PsychoPy keyboard, sub-ms RT). The target is a large outlined square with a
  thick white border. Pressing before it appears is logged as "too soon".

## Stimuli

All stimuli live under `stimuli/`. Drop your images into these folders (any
.png/.jpg/.jpeg, auto-discovered):

```
stimuli/shapes/         heart.png, circle.png, triangle.png (the bandit symbols)
stimuli/win/sweet/      sweet food images (win feedback + food bonus cue)
stimuli/win/savory/     savory food images (win feedback + food bonus cue)
stimuli/neutral/        neutral / scrambled images (neutral bonus cue)
stimuli/loss/           loss feedback images
```

Images are drawn with their aspect ratio preserved: each fits inside its display
box (the bonus cue inside a 0.8-of-height box, win and loss feedback inside
smaller boxes) without stretching, so non-square photos and scrambled images are
not distorted. Square 1024x1024 sources still work and simply fill the box.

With the `sweet+savory` food set, win feedback and the food bonus cue draw from
both folders, choosing a folder 50/50 per image and then a picture from it, so
the two folders are represented equally regardless of how many files each holds.
Neutral cues always come from `stimuli/neutral/`.

If a bandit symbol file is missing, a plain dark placeholder shape is drawn
instead, so the task runs before you add your own. Empty food/neutral/loss
folders fall back to a labelled box (cues) or a drawn sad face (losses).

## Task parameters

- Reward profiles (p_win/p_loss): 80/20 (EV +6), 50/50 (EV 0), 30/70 (EV -4).
  Symbol-to-arm and arm-to-position mappings are randomized per session, so the
  best option is not tied to any fixed symbol or screen location.
- 200 bandit trials, two reversals at trials 69 and 130. Each reversal rotates
  all three profiles in a random direction (a 3-cycle, so every arm changes
  role, including the chance arm).
- Bandit timing: 400 ms choice animation, then a jittered 400-800 ms
  anticipatory fixation (logged per trial as `anticip_ms`), 1500 ms feedback,
  then a jittered 400-700 ms ISI (logged as `isi_ms`). The pre-feedback fixation
  decorrelates choice-locked from feedback-locked responses and gives a clean
  pre-feedback baseline for the iEEG recordings.
- Bonus block: 16 food + 14 neutral, phase-stratified across the three task
  phases set by the reversals. Each phase gets an equal count of bonus trials
  (about 10) with a balanced food/neutral split (6/4, 5/5, 5/5), positions are
  spread within each phase and buffered around both reversals, and cue type is
  shuffled within phase rather than alternating, so the upcoming cue stays
  unpredictable. Sequence per bonus: "Bonus round!" (1 s), cue (1.5 s),
  anticipatory fixation jittered 1500-3000 ms, outlined square (adaptive window),
  500 ms grace, feedback (1.5 s). A hit earns 15 points.
- Adaptive window: starts 450 ms, hit shortens by 15 ms, miss lengthens by
  30 ms, floored at 250 ms and capped at 600 ms (converges near 66.7% hits). A
  press up to 500 ms after the window closes is logged as a miss but keeps its
  target RT, so slow responses still contribute a response time.

## Points

The header shows task points and bonus points, right-justified at the top of the
screen. The numeric trial counter is hidden from the participant (the progress
bar is kept); the trial number is still recorded in the data. Bonus points are
tracked on a separate tally and added to the task score for the combined TOTAL on
the end screen. Bonus rows never carry a bandit win/loss outcome, so the bandit
reward rate is unaffected.

## Output

Each run creates its own folder under `data/`, named
`sub-<pid>_ses-<ses>_<n>`, where `<n>` increments to the first unused number, so
reruns with the same ID and session never overwrite. The folder is claimed
atomically, so concurrent starts cannot collide. Inside it are a CSV and a
matching `.log`, both named after the folder.

The CSV is written one row at a time and flushed (a crash or Escape quit keeps
everything up to the last completed trial; a try/finally also closes the file on
any exit). Rows are in chronological order; `trial_type` is `bandit`,
`bonus_food`, or `bonus_neutral`. The column set matches the web export, plus
`session`, `task_version`, `t_onset_s`, `anticip_ms`, `isi_ms`, and
`trigger_code`. The `.log` records every event label with its marker for offline
alignment.

Note: `task_version` is written from the `TASK_VERSION` constant near the top of
the script (`v13` in this build) and is stamped into every data row. The `.log`
also records the staircase configuration (start, floor, ceiling, steps) once at
startup, so the floor actually used can be verified per run.

## Reproducibility

The bandit reward schedule uses the same mulberry32 generator and draw order as
the web version (symbol placement, slot order, then per-trial outcomes), so a
given seed reproduces the schedule. The lab spec differs from the web spec in
profiles and reversals (two reversals at 69 and 130, 80/20 best arm), so the
schedule matches the web version only where those settings match.

Cosmetic draws (food set, which pictures, image tilt, anticipation jitter, ISI
jitter, and the bonus schedule) run on a separate seed-derived stream and are
reproducible within PsychoPy. Two points to note: the bonus schedule is now
phase-stratified and lab-specific, so it no longer follows the web deck order;
and the `sweet+savory` food set draws an extra value per image to pick a folder,
so single-folder sets reproduce exactly as before while `sweet+savory` has its
own draw pattern.

## iEEG notes

- Triggers: every event sends the same marker, a comma. Choose `serial` (writes
  the byte `,` = 0x2C to the configured port) or `parallel` (writes the comma
  byte 44). With no device present the marker is logged only. Event identity is
  preserved in the `.log` labels and in the data file via `trial_type` and the
  onset-time columns. The event set includes the new `anticipation` marker for
  the pre-feedback fixation. If your recording system needs a different transport
  or a distinct code per event, that is a small change in the `Triggers` class
  and `EVENT_CODES`.
- Photodiode: a white square pulses bottom-right at every event onset (choice,
  anticipatory fixation, outcome, cue, target square, feedback). Reposition or
  resize `pd_stim` for your sensor.
- Timing is frame-locked; onset timestamps in the data are flip times.