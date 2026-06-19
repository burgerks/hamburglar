#!/usr/bin/env python3
"""
Probabilistic Bandit Task with embedded craving (wanting) ratings  -  PsychoPy port.

3-arm probabilistic reversal-learning task. Three reward profiles, one per arm:
    profile A: 70% reward (+10), 30% loss (-10)   (EV +4, best)
    profile B: 30% reward (+10), 70% loss (-10)   (EV -4, worst)
    profile C: 50% reward (+10), 50% loss (-10)   (EV  0, chance)
Outcomes are binary (+10 / -10). A single reversal at a fixed trial cyclically
rotates all three profiles in a random direction, so which arm is best flips.

Per session a random seed is drawn and logged in every row. That seed reproduces
the outcome sequence, the reversal direction, the symbol-to-arm assignment, the
screen placement, and the win/loss image order. The reversal trial and the
craving-rating schedule are fixed for every participant.

Input is the keyboard arrows: Left = left symbol, Down = middle, Right = right.
Feedback is a 1 s picture (win or loss) centered at ~50% of screen width with the
signed points printed below it. Win pictures are drawn from feedback/win/<version>
where <version> is "sweet" or "savory", chosen in the start dialog. Loss pictures
are drawn from feedback/loss. Within each set the script cycles through all images
before any repeat.

Craving rating: a 4 s button-mash on a 0-20 thermometer that starts in the middle.
Right arrow raises the level, Left lowers it, and the live press count is shown.
Ten ratings are logged (one before the first trial, then nine more 15-25 trials
apart, with one immediately before the reversal), plus one unlogged practice rating.

Practice: 5 unlogged bandit trials, all arms 50/50, on a separate RNG so practice
never reveals the best symbol and never perturbs the real schedule.

Output: one long-format CSV (one row per logged event, event_type = choice or
wanting), written to a data/ folder next to this script.

Requires PsychoPy:  pip install psychopy
Folder layout next to this script:
    feedback/win/sweet/   (numbered images, e.g. 1.jpg, 2.jpg, ...)
    feedback/win/savory/  (numbered images)
    feedback/loss/        (numbered images)
"""

import os
import re
import csv
import glob
import random
import datetime

from psychopy import visual, core, gui
from psychopy.hardware import keyboard
from PIL import Image

# ===================================================================
#  Configuration
# ===================================================================
CFG = dict(
    N_ARMS=3,
    N_TRIALS=200,
    REWARD_PTS=10,
    LOSS_PTS=-10,
    # Arm probability profiles [p_reward, p_loss]
    PROFILE_A=[0.70, 0.30],   # best   (EV +4)
    PROFILE_B=[0.30, 0.70],   # worst  (EV -4)
    PROFILE_C=[0.50, 0.50],   # chance (EV  0)
    REVERSAL_TRIAL=112,       # 1-indexed trial at which the single reversal applies
    # Practice (not logged, separate RNG, all arms 50/50)
    N_PRACTICE=5,
    PRACTICE_PROFILE=[0.50, 0.50],
    # Craving ("wanting") rating
    CRAVING_MAX=20,           # full thermometer = value 20
    CRAVING_START=10,         # hash starts in the middle
    CRAVING_MS=4000,          # fixed 4 s response window
    # Trials BEFORE which a logged craving rating appears (112 = just before the
    # reversal; gaps run 15-25 trials). Fixed for every participant.
    WANTING_BEFORE_TRIALS=[1, 18, 40, 58, 78, 95, 112, 133, 154, 175],
    # Timing (ms)
    FEEDBACK_MS=1000,         # win/loss picture duration
    ITI_MS=500,
    ANIM_MS=350,              # selection-confirmation highlight
    FULLSCREEN=True,
)

# Symbol identities for the three logical arms (drawn as shapes, asset-free).
SHAPES = ["circle", "triangle", "square"]

# Colors as rgb255 tuples (PsychoPy colorSpace='rgb255').
COL = dict(
    bg=(22, 22, 30), bg2=(26, 27, 38), fg=(255, 255, 255), dim=(139, 149, 184),
    accent=(122, 162, 247), gold=(224, 175, 104), green=(158, 206, 106),
    red=(247, 118, 142), box=(251, 251, 253), symbol=(40, 50, 90),
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")


class QuitTask(Exception):
    """Raised when the experimenter presses q or escape to abort and save."""
    pass


# ===================================================================
#  Start dialog: participant ID + food version (dropdown)
# ===================================================================
def get_session_info():
    """Show a startup dialog; the list value becomes a sweet/savory dropdown."""
    info = {"participant_id": "", "food_version": ["sweet", "savory"]}
    dlg = gui.DlgFromDict(info, title="Probabilistic Bandit Task",
                          order=["participant_id", "food_version"])
    if not dlg.OK:
        core.quit()
    info["participant_id"] = str(info["participant_id"]).strip() or "NA"
    return info


# ===================================================================
#  Image discovery and cycling
# ===================================================================
def _num_key(path):
    """Sort key from the first integer in the filename, else fall back to name."""
    m = re.search(r"\d+", os.path.basename(path))
    return (0, int(m.group())) if m else (1, os.path.basename(path).lower())


def list_images(folder):
    """Return image files in a folder, sorted by the number in their filename."""
    files = [f for f in glob.glob(os.path.join(folder, "*"))
             if f.lower().endswith(IMG_EXTS)]
    return sorted(files, key=_num_key)


def make_cycler(files, rng):
    """Yield images in shuffled passes: exhaust the whole set before any repeat."""
    order = []

    def nxt():
        if not order:
            order.extend(files)
            rng.shuffle(order)
        return order.pop()
    return nxt


def resolve_feedback_folders(version):
    """Build and validate the win/loss image folders next to this script."""
    win_dir = os.path.join(SCRIPT_DIR, "feedback", "win", version)
    loss_dir = os.path.join(SCRIPT_DIR, "feedback", "loss")
    win_imgs, loss_imgs = list_images(win_dir), list_images(loss_dir)
    problems = []
    if not win_imgs:
        problems.append(f"No win images found in:\n{win_dir}")
    if not loss_imgs:
        problems.append(f"No loss images found in:\n{loss_dir}")
    if problems:
        dlg = gui.Dlg(title="Missing feedback images")
        for p in problems:
            dlg.addText(p)
        dlg.addText("Add numbered images to these folders and run again.")
        dlg.show()
        core.quit()
    return win_imgs, loss_imgs


# ===================================================================
#  RNG and reward schedule
# ===================================================================
def make_rngs():
    """Main RNG reproduces the schedule from SEED; image and practice RNGs are
    seeded separately so they do not shift the main draw order."""
    seed = random.randrange(2 ** 32)
    return seed, random.Random(seed), random.Random(seed ^ 0x9E3779B9), \
        random.Random(random.randrange(2 ** 32))


def shuffle3(rng):
    """Return a random permutation of [0, 1, 2] from the given RNG."""
    a = [0, 1, 2]
    rng.shuffle(a)
    return a


def sample_outcome(profile, rng):
    """Draw a single binary outcome from a [p_reward, p_loss] profile."""
    return "reward" if rng.random() < profile[0] else "loss"


# ===================================================================
#  Session state
# ===================================================================
class State:
    """Holds everything that changes during a run plus the logged-event list."""

    def __init__(self, info, seed):
        self.pid = info["participant_id"]
        self.version = info["food_version"]
        self.seed = seed
        self.trial = 0                 # 0-indexed bandit trial
        self.score = 0
        self.swap_count = 0
        self.profiles = [list(CFG["PROFILE_A"]), list(CFG["PROFILE_B"]),
                         list(CFG["PROFILE_C"])]
        self.symbol_map = None         # logical arm -> shape index
        self.slot_order = None         # screen position -> logical arm
        self.event_index = 0           # 1-based over all logged events
        self.last_craving = ""         # forward-filled onto choice rows
        self.first_abs = None          # absolute time of the first logged event
        self.rows = []                 # logged event rows

    def rel_ms(self, abs_t):
        """ms from task start; the first call sets t = 0 (first logged event)."""
        if self.first_abs is None:
            self.first_abs = abs_t
        return round((abs_t - self.first_abs) * 1000)


def apply_reversal_if_needed(S, rng):
    """At the reversal trial, rotate all three profiles in a random direction."""
    if S.trial == CFG["REVERSAL_TRIAL"] - 1:
        p = S.profiles
        S.profiles = [p[2], p[0], p[1]] if rng.random() < 0.5 else [p[1], p[2], p[0]]
        S.swap_count += 1


# ===================================================================
#  CSV output
# ===================================================================
COLUMNS = [
    "participant_id", "food_version", "seed", "event_index", "event_type", "phase",
    "onset_ms", "duration_ms",
    # choice-specific
    "trial", "swap_count", "position1", "position2", "position3",
    "p_reward_pos1", "p_reward_pos2", "p_reward_pos3",
    "choice", "chosen_logo", "response_key", "rt_ms", "rt_s", "outcome", "points",
    "optimal_position", "is_optimal", "optimal_points", "regret", "cumulative_score",
    "choice_onset_ms", "feedback_onset_ms", "feedback_image", "last_craving",
    "cf_outcome_pos1", "cf_outcome_pos2", "cf_outcome_pos3",
    "cf_points_pos1", "cf_points_pos2", "cf_points_pos3",
    # wanting-specific
    "wanting_index", "wanting_before_trial", "craving_start", "craving_rating",
    "craving_pct", "n_right", "n_left", "n_press_total", "first_press_ms",
    # appended to every row
    "task_duration_s",
]


def write_csv(S):
    """Write all logged rows to data/<pid>_<version>_<timestamp>.csv."""
    if not S.rows:
        return None
    out_dir = os.path.join(SCRIPT_DIR, "data")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_pid = re.sub(r"[^A-Za-z0-9_-]", "_", S.pid)
    path = os.path.join(out_dir, f"prob_bandit_{safe_pid}_{S.version}_{ts}.csv")
    dur = round((core.getTime() - S.first_abs), 2) if S.first_abs is not None else ""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in S.rows:
            row = dict(r)
            row["task_duration_s"] = dur
            w.writerow(row)
    return path


# ===================================================================
#  Stimuli
# ===================================================================
def txt(win, text, **kw):
    """TextStim helper with white-on-dark defaults in height units."""
    kw.setdefault("color", COL["fg"])
    kw.setdefault("colorSpace", "rgb255")
    kw.setdefault("height", 0.045)
    kw.setdefault("wrapWidth", 1.5)
    return visual.TextStim(win, text=text, **kw)


def build_symbols(win):
    """Create the three arm symbols once; they are repositioned each trial."""
    return {
        "circle": visual.Circle(win, radius=0.075, edges=64,
                                fillColor=COL["symbol"], lineColor=COL["symbol"],
                                colorSpace="rgb255"),
        "triangle": visual.Polygon(win, edges=3, radius=0.095,
                                   fillColor=COL["symbol"], lineColor=COL["symbol"],
                                   colorSpace="rgb255"),
        "square": visual.Rect(win, width=0.14, height=0.14,
                              fillColor=COL["symbol"], lineColor=COL["symbol"],
                              colorSpace="rgb255"),
    }


def fit_image_size(win, path):
    """Target size (height units) so the image is ~50% screen width and not clipped."""
    iw, ih = Image.open(path).size
    img_aspect = iw / ih
    win_aspect = win.size[0] / win.size[1]
    max_w, max_h = 0.5 * win_aspect, 0.52
    w = max_w
    h = w / img_aspect
    if h > max_h:
        h, w = max_h, max_h * img_aspect
    return (w, h)


# ===================================================================
#  Low-level input helpers
# ===================================================================
def check_quit(keys):
    """Raise QuitTask if q or escape appears in a list of key presses."""
    for k in keys:
        if k.name in ("escape", "q"):
            raise QuitTask()


def wait_with_quit(win, kb, seconds):
    """Hold the current frame for a fixed time while still allowing quit keys."""
    clock = core.Clock()
    while clock.getTime() < seconds:
        check_quit(kb.getKeys(["escape", "q"], waitRelease=False))
        core.wait(0.002)


# ===================================================================
#  Screens
# ===================================================================
def show_text(win, kb, body):
    """Draw an instruction screen and wait for space or enter to continue."""
    block = txt(win, body, pos=(0, 0.06), height=0.042, wrapWidth=1.5,
                alignText="left", anchorHoriz="center")
    hint = txt(win, "Press SPACE to continue", pos=(0, -0.42), height=0.035,
               color=COL["dim"])
    block.draw()
    hint.draw()
    win.flip()
    kb.clearEvents()
    while True:
        keys = kb.getKeys(["space", "return", "escape", "q"], waitRelease=False)
        check_quit(keys)
        if any(k.name in ("space", "return") for k in keys):
            return
        core.wait(0.002)


def draw_header(win, S):
    """Top status line shown during the real task (hidden during practice)."""
    txt(win, f"Trial {S.trial + 1} / {CFG['N_TRIALS']}", pos=(-0.55, 0.45),
        height=0.035, color=COL["dim"], anchorHoriz="left", alignText="left").draw()
    txt(win, f"Score: {S.score} pts", pos=(0.55, 0.45), height=0.035,
        color=COL["gold"], anchorHoriz="right", alignText="right").draw()


def draw_choice(win, S, symbols, boxes, prompt, is_practice,
                chosen_pos=None, highlight=None):
    """Render the three symbols; optionally highlight the chosen one and dim the rest."""
    if not is_practice:
        draw_header(win, S)
    prompt.draw()
    box_x = [-0.38, 0.0, 0.38]
    for pos in range(3):
        arm = S.slot_order[pos]
        shape = symbols[SHAPES[S.symbol_map[arm]]]
        dim = chosen_pos is not None and pos != chosen_pos
        boxes[pos].opacity = 0.4 if dim else 1.0
        shape.opacity = 0.4 if dim else 1.0
        boxes[pos].pos = (box_x[pos], 0)
        shape.pos = (box_x[pos], 0)
        boxes[pos].draw()
        shape.draw()
        if chosen_pos == pos and highlight is not None:
            highlight.pos = (box_x[pos], 0)
            highlight.draw()


def run_choice(win, kb, S, symbols, boxes, highlight, is_practice):
    """Self-paced choice via Left/Down/Right; brief confirm, then return the choice."""
    key_pos = {"left": 0, "down": 1, "right": 2}
    prompt = txt(win, ("Practice. " if is_practice else "") +
                 "Use  <  v  >  to choose a symbol",
                 pos=(0, 0.30), height=0.045)
    draw_choice(win, S, symbols, boxes, prompt, is_practice)
    onset = win.flip()
    kb.clock.reset()
    kb.clearEvents()

    chosen_pos, key_name, rt_ms = None, None, None
    while chosen_pos is None:
        keys = kb.getKeys(["left", "down", "right", "escape", "q"], waitRelease=False)
        check_quit(keys)
        for k in keys:
            if k.name in key_pos:
                chosen_pos, key_name, rt_ms = key_pos[k.name], k.name, k.rt * 1000.0
                break
        core.wait(0.002)

    # Brief selection-confirmation highlight, then advance.
    draw_choice(win, S, symbols, boxes, prompt, is_practice,
                chosen_pos=chosen_pos, highlight=highlight)
    win.flip()
    wait_with_quit(win, kb, CFG["ANIM_MS"] / 1000.0)

    arm = S.slot_order[chosen_pos]
    return dict(pos=chosen_pos, arm=arm, key=key_name, rt_ms=rt_ms, onset=onset)




def show_feedback(win, kb, S, ch, rng, img_cache, win_next, loss_next,
                  is_practice, practice_rng=None):
    """1 s picture feedback; samples the outcome and logs the choice row (real only)."""
    if is_practice:
        # Practice: 50/50 via the practice RNG; not logged.
        outcome = "reward" if practice_rng.random() < CFG["PRACTICE_PROFILE"][0] else "loss"
        points = CFG["REWARD_PTS"] if outcome == "reward" else CFG["LOSS_PTS"]
        img_path = win_next() if outcome == "reward" else loss_next()
        cf_out = cf_pts = arm = None
    else:
        # Real trial: one RNG draw (reversal check, then sample all three arms),
        # then pick the picture that matches the realized outcome.
        apply_reversal_if_needed(S, rng)
        cf_out = [sample_outcome(p, rng) for p in S.profiles]
        cf_pts = [CFG["REWARD_PTS"] if o == "reward" else CFG["LOSS_PTS"] for o in cf_out]
        arm = ch["arm"]
        outcome, points = cf_out[arm], cf_pts[arm]
        img_path = win_next() if outcome == "reward" else loss_next()

    # Build / fetch the sized image and draw it with the signed points below.
    if img_path not in img_cache:
        img_cache[img_path] = visual.ImageStim(win, image=img_path,
                                               size=fit_image_size(win, img_path),
                                               pos=(0, 0.12))
    pts_color = COL["green"] if outcome == "reward" else COL["red"]
    sign = "+" if points > 0 else ""
    pts = txt(win, f"{sign}{points} points", pos=(0, -0.36), height=0.07,
              color=pts_color, bold=True)
    img_cache[img_path].draw()
    pts.draw()
    fb_onset = win.flip()

    if not is_practice:
        # Log the row now that the feedback onset is known.
        S.score += points
        _log_choice_row(S, ch, outcome, points, cf_out, cf_pts, arm, fb_onset, img_path)

    wait_with_quit(win, kb, CFG["FEEDBACK_MS"] / 1000.0)


def _log_choice_row(S, ch, outcome, points, cf_out, cf_pts, arm, fb_onset, img_path):
    """Append a fully populated choice row (helper for show_feedback)."""
    L = lambda p: S.slot_order[p - 1]
    sym = lambda a: SHAPES[S.symbol_map[a]]
    choice_pos = ch["pos"] + 1
    p_rewards = [p[0] for p in S.profiles]
    opt_arm = p_rewards.index(max(p_rewards))
    optimal_pos = S.slot_order.index(opt_arm) + 1
    optimal_points = cf_pts[opt_arm]

    S.event_index += 1
    S.rows.append(dict(
        participant_id=S.pid, food_version=S.version, seed=S.seed,
        event_index=S.event_index, event_type="choice", phase="main",
        onset_ms=S.rel_ms(ch["onset"]), duration_ms=round(ch["rt_ms"]),
        trial=S.trial + 1, swap_count=S.swap_count,
        position1=sym(L(1)), position2=sym(L(2)), position3=sym(L(3)),
        p_reward_pos1=round(S.profiles[L(1)][0], 4),
        p_reward_pos2=round(S.profiles[L(2)][0], 4),
        p_reward_pos3=round(S.profiles[L(3)][0], 4),
        choice=choice_pos, chosen_logo=sym(arm), response_key=ch["key"],
        rt_ms=round(ch["rt_ms"]), rt_s=round(ch["rt_ms"] / 1000.0, 4),
        outcome=outcome, points=points,
        optimal_position=optimal_pos, is_optimal=int(choice_pos == optimal_pos),
        optimal_points=optimal_points, regret=optimal_points - points,
        cumulative_score=S.score,
        choice_onset_ms=S.rel_ms(ch["onset"]),
        feedback_onset_ms=S.rel_ms(fb_onset),
        feedback_image=os.path.basename(img_path),
        last_craving=S.last_craving,
        cf_outcome_pos1=cf_out[L(1)], cf_outcome_pos2=cf_out[L(2)],
        cf_outcome_pos3=cf_out[L(3)],
        cf_points_pos1=cf_pts[L(1)], cf_points_pos2=cf_pts[L(2)],
        cf_points_pos3=cf_pts[L(3)],
    ))
    S.trial += 1


def run_iti(win, kb):
    """Blank inter-trial interval."""
    win.flip()
    wait_with_quit(win, kb, CFG["ITI_MS"] / 1000.0)


def run_wanting(win, kb, S, is_practice, wanting_index="", before_trial=""):
    """4 s craving rating. Hash starts in the middle; Right raises, Left lowers."""
    value = CFG["CRAVING_START"]
    n_right = n_left = 0
    first_press = ""
    track_w, track_x0 = 0.9, -0.45                       # full bar geometry
    cmax = CFG["CRAVING_MAX"]

    question = txt(win, "How much are you craving food?", pos=(0, 0.30),
                   height=0.05, bold=True)
    track = visual.Rect(win, width=track_w, height=0.07, pos=(0, 0),
                        fillColor=COL["bg2"], lineColor=COL["dim"],
                        colorSpace="rgb255")
    fill = visual.Rect(win, width=0.001, height=0.07, pos=(track_x0, 0),
                       fillColor=COL["accent"], lineColor=None, colorSpace="rgb255")
    hash_mark = visual.Rect(win, width=0.008, height=0.11, pos=(0, 0),
                            fillColor=COL["gold"], lineColor=None, colorSpace="rgb255")
    # Anchors stacked one word per line, centered under each end of the bar.
    left_anchor = txt(win, "I'm\nnot\ninterested\nin\nfood", pos=(track_x0, -0.20),
                      height=0.034, color=COL["dim"], wrapWidth=0.4)
    right_anchor = txt(win, "Biggest\nimaginable\ncraving", pos=(track_x0 + track_w, -0.20),
                       height=0.034, color=COL["dim"], wrapWidth=0.4)
    hint = txt(win, "Press  >  to raise,  <  to lower", pos=(0, -0.40),
               height=0.035, color=COL["dim"])
    presses = txt(win, "Presses: 0", pos=(0, 0.18), height=0.04, color=COL["fg"])
    timer = visual.Rect(win, width=track_w, height=0.012, pos=(0, -0.46),
                        fillColor=COL["dim"], lineColor=None, colorSpace="rgb255")
    tag = txt(win, "Practice", pos=(0, -0.33), height=0.035, color=COL["gold"])

    def draw_all(remaining_frac):
        frac = value / cmax
        fw = max(0.001, frac * track_w)            # left edge stays fixed at track_x0
        fill.width = fw
        fill.pos = (track_x0 + fw / 2.0, 0)
        hash_mark.pos = (track_x0 + frac * track_w, 0)
        presses.text = f"Presses: {n_right + n_left}"
        tw = max(0.001, remaining_frac * track_w)
        timer.width = tw
        timer.pos = (track_x0 + tw / 2.0, -0.46)
        question.draw(); track.draw(); fill.draw(); hash_mark.draw()
        left_anchor.draw(); right_anchor.draw(); presses.draw(); hint.draw()
        timer.draw()
        if is_practice:
            tag.draw()

    draw_all(1.0)
    onset = win.flip()
    kb.clock.reset()
    kb.clearEvents()

    dur = CFG["CRAVING_MS"] / 1000.0
    clock = core.Clock()
    while clock.getTime() < dur:
        keys = kb.getKeys(["left", "right", "escape", "q"], waitRelease=False)
        check_quit(keys)
        for k in keys:
            if k.name == "right":
                value = min(cmax, value + 1)
                n_right += 1
                if first_press == "":
                    first_press = round(k.rt * 1000.0)
            elif k.name == "left":
                value = max(0, value - 1)
                n_left += 1
                if first_press == "":
                    first_press = round(k.rt * 1000.0)
        draw_all(max(0.0, 1.0 - clock.getTime() / dur))
        win.flip()

    if not is_practice:
        S.last_craving = value
        S.event_index += 1
        S.rows.append(dict(
            participant_id=S.pid, food_version=S.version, seed=S.seed,
            event_index=S.event_index, event_type="wanting", phase="main",
            onset_ms=S.rel_ms(onset), duration_ms=CFG["CRAVING_MS"],
            wanting_index=wanting_index, wanting_before_trial=before_trial,
            craving_start=CFG["CRAVING_START"], craving_rating=value,
            craving_pct=round(value / cmax * 100), n_right=n_right, n_left=n_left,
            n_press_total=n_right + n_left, first_press_ms=first_press,
        ))


# ===================================================================
#  Main
# ===================================================================
def main():
    info = get_session_info()
    win_imgs, loss_imgs = resolve_feedback_folders(info["food_version"])
    seed, rng_main, rng_img, rng_prac = make_rngs()

    S = State(info, seed)
    S.symbol_map = shuffle3(rng_main)      # logical arm -> shape  (reproducible)
    S.slot_order = shuffle3(rng_main)      # screen position -> arm (reproducible)

    # Image cyclers: main task is reproducible from the seed; practice is separate.
    win_next, loss_next = make_cycler(win_imgs, rng_img), make_cycler(loss_imgs, rng_img)
    pwin_next, ploss_next = make_cycler(win_imgs, rng_prac), make_cycler(loss_imgs, rng_prac)

    win = visual.Window(fullscr=CFG["FULLSCREEN"], color=COL["bg"],
                        colorSpace="rgb255", units="height", allowGUI=False)
    kb = keyboard.Keyboard()
    symbols = build_symbols(win)
    boxes = [visual.Rect(win, width=0.26, height=0.26, fillColor=COL["box"],
                         lineColor=COL["dim"], colorSpace="rgb255") for _ in range(3)]
    highlight = visual.Rect(win, width=0.30, height=0.30, fillColor=None,
                            lineColor=COL["accent"], lineWidth=6, colorSpace="rgb255")
    img_cache = {}

    try:
        show_text(win, kb,
                  "In this task you will see three symbols. You choose one with the "
                  "keyboard arrows: the LEFT arrow picks the left symbol, the DOWN "
                  "arrow picks the middle symbol, and the RIGHT arrow picks the right "
                  "symbol. This is a hard task, so go with your gut.\n\n"
                  "After you choose, a picture appears and you win 10 points or lose "
                  "10 points. Your goal is to earn as many points as you can.\n\n"
                  "The three symbols do not pay off equally, none wins every time, and "
                  "the best symbol can change part way through. Keep tracking how each "
                  "one is doing.\n\n"
                  "We will start with a few practice trials. You can press q to quit at "
                  "any time.")

        # Practice bandit trials (not logged, all arms 50/50, separate RNG).
        for _ in range(CFG["N_PRACTICE"]):
            ch = run_choice(win, kb, S, symbols, boxes, highlight, is_practice=True)
            show_feedback(win, kb, S, ch, None, img_cache, pwin_next, ploss_next,
                          is_practice=True, practice_rng=rng_prac)
            run_iti(win, kb)

        # Craving-rating instructions + one practice rating.
        show_text(win, kb,
                  "Once in a while the task will pause and ask how much you are craving "
                  "food right now.\n\n"
                  "You answer with a bar that starts in the middle. Press the RIGHT "
                  "arrow to raise your craving level and the LEFT arrow to lower it. "
                  "The far left means you are not interested in food, and the far right "
                  "means the biggest craving you can imagine.\n\n"
                  "You have a few seconds for each rating. Let's try one for practice.")
        run_wanting(win, kb, S, is_practice=True)
        run_iti(win, kb)

        # Transition into the real task.
        show_text(win, kb,
                  "That is the end of practice. The real task starts now and your "
                  "points will count.\n\n"
                  "Choose with the LEFT, DOWN, and RIGHT arrows for the left, middle, "
                  "and right symbols, and keep tracking which symbol is paying off.\n\n"
                  "We will begin with a quick craving rating.")

        # Main timeline: 200 trials with logged ratings inserted before scheduled trials.
        for i in range(1, CFG["N_TRIALS"] + 1):
            if i in CFG["WANTING_BEFORE_TRIALS"]:
                idx = CFG["WANTING_BEFORE_TRIALS"].index(i) + 1
                run_wanting(win, kb, S, is_practice=False, wanting_index=idx,
                            before_trial=i)
                run_iti(win, kb)
            ch = run_choice(win, kb, S, symbols, boxes, highlight, is_practice=False)
            show_feedback(win, kb, S, ch, rng_main, img_cache, win_next, loss_next,
                          is_practice=False)
            run_iti(win, kb)

        # End screen.
        n = sum(1 for r in S.rows if r["event_type"] == "choice")
        show_text(win, kb,
                  f"Great work. You finished the task.\n\n"
                  f"Final score: {S.score} points\n"
                  f"Trials completed: {n}\n\n"
                  "Your responses have been saved. Thank you.")

    except QuitTask:
        pass
    finally:
        path = write_csv(S)
        win.close()
        if path:
            print(f"Saved data to: {path}")
        core.quit()


if __name__ == "__main__":
    main()
