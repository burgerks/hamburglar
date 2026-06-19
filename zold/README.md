# Bandit + mini-MID bonus (PsychoPy)

PsychoPy port of the jsPsych task. Same reward schedule, single reversal at
trial 112, adaptive-window bonus block (16 food + 14 neutral), and data fields.

## Run

```
python bandit_mid_task.py
```

A startup dialog collects participant ID, session, an optional seed (blank draws
a random one, logged in every row), the food set (auto/sweet/savory), and the
iEEG options (photodiode square, parallel-port triggers + address). Press SPACE
at the instructions to begin. The experimenter can abort at any time with
Escape; data written so far is kept.

Requires PsychoPy (2023.2 or newer recommended). Install via the standalone
PsychoPy app or `pip install psychopy`.

## Stimuli

Drop your images into these folders (any .png/.jpg/.jpeg, auto-discovered):

```
stimuli/win/sweet/      sweet food images
stimuli/win/savory/     savory food images
stimuli/neutral cues/   neutral object images
```

`stimuli/symbols/{heart,circle,triangle}.png` are included. Empty folders fall
back to a labelled placeholder so the task still runs for piloting.

## Output

One CSV per session in `data/`, written one row at a time and flushed (a crash
mid-session keeps everything up to the last completed trial). Rows are in
chronological order; `trial_type` is `bandit`, `bonus_food`, or `bonus_neutral`.
The column set matches the web export, plus `session`, `t_onset_s`, and
`trigger_code`. A matching `.log` records trigger codes for offline alignment.

## Reproducibility

The bandit reward schedule uses the same mulberry32 generator and the same draw
order as the web version, so a given seed produces the same schedule (verified
bit-for-bit against the JavaScript). Cosmetic draws (food set, which pictures,
corner tilt, bonus placement) run on a separate seed-derived stream and are
reproducible within PsychoPy.

## iEEG notes

- Photodiode: a white square pulses bottom-right at every event onset (cue,
  square, outcome). Reposition or resize `pd_stim` in `build_window`/`main` for
  your sensor.
- Triggers: choose `parallel` in the dialog and set the port address. Codes are
  in `EVENT_CODES` near the top of the script. With no port present the codes
  still go to the `.log`.
- Timing is frame-locked; onset timestamps in the data are flip times.

## Decisions worth a look

- Bandit choice is a mouse click (as in the web task); the bonus target is any
  keyboard key (PsychoPy keyboard, sub-ms RT). Say if you want a button box or
  keyboard choices instead.
- The web inactivity auto-abort was dropped (an experimenter is present); the
  per-trial 4 s "Please answer faster." nudge is kept.
- Loss feedback draws a font-independent sad face (shapes), not an emoji glyph,
  so it renders the same on any machine.
- Bonus points are a separate tally and stay out of the bandit score and the
  win/loss counts.
