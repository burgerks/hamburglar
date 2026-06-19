#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Probabilistic 3-arm bandit with an interleaved mini-MID "bonus round".

PsychoPy (Coder) port of the jsPsych task. Same reward schedule, same single
reversal, same adaptive-window bonus, same data fields. The mulberry32 RNG is
reproduced bit-for-bit from JavaScript, so a given seed yields the same bandit
schedule as the web version (verified against the JS implementation).

Tested against the PsychoPy 2023.2+/2024.x API (visual, core, event, gui,
hardware.keyboard, parallel). Run from the PsychoPy Coder or `python bandit_mid_task.py`.

Folder layout expected next to this file:
    stimuli/symbols/{heart,circle,triangle}.png   (provided)
    stimuli/win/sweet/      <your sweet food images>
    stimuli/win/savory/     <your savory food images>
    stimuli/neutral cues/   <your neutral object images>
Images are auto-discovered (any .png/.jpg/.jpeg). Empty folders fall back to a
labelled placeholder so the task still runs for piloting.
"""

import os
import csv
import glob
import math
import datetime

from psychopy import visual, core, event, gui, logging
from psychopy.hardware import keyboard

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG  (timing in ms, mirrors the web CFG)
# ════════════════════════════════════════════════════════════════════════════
CFG = dict(
    N_ARMS=3,
    N_TRIALS=200,
    REWARD_PTS=10,
    LOSS_PTS=-10,
    # Arm profiles [p_reward, p_loss]
    PROFILE_A=[0.70, 0.30],   # best   (EV +4)
    PROFILE_B=[0.30, 0.70],   # worst  (EV -4)
    PROFILE_C=[0.50, 0.50],   # chance (EV  0)
    REVERSAL_TRIAL=112,       # 1-indexed data trial where the single reversal takes effect

    # Bandit timing
    FEEDBACK_MS=900,
    ITI_MS=500,
    ANIM_MS=400,
    CHOICE_DEADLINE_MS=4000,  # nudge prompt after 4 s; trial still waits for a click

    # Mini-MID bonus block
    N_BONUS_FOOD=16,
    N_BONUS_NEUTRAL=14,
    BONUS_INTRO_MS=1000,
    CUE_MS=1500,
    DELAY_MIN_MS=1500,
    DELAY_MAX_MS=3500,
    GRACE_MS=500,             # late press logged as miss-with-RT, not no_response
    BONUS_FEEDBACK_MS=1500,
    BONUS_PTS=10,
    # Adaptive response-window staircase (weighted up/down ~66% hits)
    WIN_START=400,
    WIN_FLOOR=250,
    WIN_CEIL=500,
    WIN_STEP_DOWN=15,
    WIN_STEP_UP=30,
    # Bonus placement across the bandit stream
    BONUS_FIRST_AFTER=8,
    BONUS_REV_BUFFER=3,
    BONUS_MIN_GAP=3,
)

# Display / hardware (overridable in the startup dialog)
FULLSCREEN = True
BG_COLOR = [-0.5, -0.5, -0.5]   # dark grey, PsychoPy [-1..1] RGB
WIN_SIZE = [1280, 800]          # used only when FULLSCREEN is False

SYMBOL_DIR = os.path.join('stimuli', 'symbols')
IMG_DIRS = dict(
    sweet=os.path.join('stimuli', 'win', 'sweet'),
    savory=os.path.join('stimuli', 'win', 'savory'),
    neutral=os.path.join('stimuli', 'neutral cues'),
)
SYMBOL_NAMES = ['heart', 'circle', 'triangle']   # logical symbol order (matches the web)

# Trigger codes per event (parallel port). 0 is reserved for the line-clear state.
EVENT_CODES = dict(
    choice_onset=10, choice_made=11,
    bandit_win=20, bandit_loss=21,
    bonus_intro=30, cue_food=31, cue_neutral=32, fixation=33,
    target=40, response=41, bonus_feedback=42,
)


# ════════════════════════════════════════════════════════════════════════════
#  mulberry32 RNG  (two seed-derived streams)
# ════════════════════════════════════════════════════════════════════════════
# The bandit schedule rides on the MAIN stream; its call order must not change.
# All cosmetic randomness (food set, which pictures, corner tilt, bonus deck and
# placement) rides on a SECOND stream so adding stimuli never shifts the schedule.
def _imul(a, b):
    """32-bit integer multiply matching JS Math.imul (low 32 bits)."""
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) & 0xFFFFFFFF


def make_rng(seed):
    """Return a mulberry32 generator (float in [0,1)) seeded with a uint32."""
    a = seed & 0xFFFFFFFF

    def rng():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return rng


# ════════════════════════════════════════════════════════════════════════════
#  Startup dialog
# ════════════════════════════════════════════════════════════════════════════
# Collect participant info and session options. A blank seed draws a fresh random
# uint32 (logged in every row); a fixed seed reproduces a schedule exactly.
def run_dialog():
    info = {
        'participant': '',
        'session': '001',
        'seed (blank = random)': '',
        'food_set': ['auto', 'sweet', 'savory'],
        'photodiode': False,
        'triggers': ['none', 'parallel'],
        'parallel_address (hex)': '0x378',
    }
    ok = gui.DlgFromDict(info, title='Bandit + Bonus',
                         order=['participant', 'session', 'seed (blank = random)',
                                'food_set', 'photodiode', 'triggers',
                                'parallel_address (hex)'])
    if not ok.OK:
        core.quit()
    seed_txt = str(info['seed (blank = random)']).strip()
    seed = (int(seed_txt, 0) & 0xFFFFFFFF) if seed_txt else \
        (int.from_bytes(os.urandom(4), 'little'))
    return dict(
        pid=str(info['participant']).strip() or 'test',
        session=str(info['session']).strip() or '001',
        seed=seed,
        food_override=(None if info['food_set'] == 'auto' else info['food_set']),
        photodiode=bool(info['photodiode']),
        trig_mode=info['triggers'],
        parallel_addr=int(str(info['parallel_address (hex)']), 0),
    )


# ════════════════════════════════════════════════════════════════════════════
#  Triggers (parallel port, with a no-op fallback that logs)
# ════════════════════════════════════════════════════════════════════════════
# A thin wrapper so the same send()/clear() calls work whether or not a parallel
# port is present. Codes are also written to the PsychoPy log for offline checks.
class Triggers:
    def __init__(self, mode='none', address=0x378):
        self.mode = mode
        self.port = None
        if mode == 'parallel':
            try:
                from psychopy import parallel
                self.port = parallel.ParallelPort(address=address)
                self.port.setData(0)
            except Exception as e:
                logging.warn('Parallel port unavailable (%s); triggers logged only.' % e)
                self.mode = 'none'

    def send(self, code):
        if self.port is not None:
            self.port.setData(int(code) & 0xFF)
        logging.exp('TRIGGER %d' % code)

    def clear(self):
        if self.port is not None:
            self.port.setData(0)


# ════════════════════════════════════════════════════════════════════════════
#  Reward schedule  (reproduces the web logic and rand() call order)
# ════════════════════════════════════════════════════════════════════════════
def shuffle3(rand):
    """Fisher-Yates permutation of [0,1,2] using the MAIN stream (2 rand calls)."""
    a = [0, 1, 2]
    for i in range(len(a) - 1, 0, -1):
        j = int(rand() * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a


def sample_outcome(rand, profile):
    """Draw 'reward' or 'loss' from a [p_reward, p_loss] profile (1 rand call)."""
    return 'reward' if rand() < profile[0] else 'loss'


def apply_reversal(rand, profiles, zero_idx, swap_trial_idx):
    """At the reversal trial, rotate all three profiles in a random direction
    (1 rand call). A 3-cycle has no fixed point, so every arm changes role."""
    if zero_idx == swap_trial_idx:
        p = profiles
        if rand() < 0.5:
            return [p[2], p[0], p[1]], True   # rotate right
        return [p[1], p[2], p[0]], True       # rotate left
    return profiles, False


# ════════════════════════════════════════════════════════════════════════════
#  Cosmetic helpers (SECOND stream): food set, image draws, bonus schedule
# ════════════════════════════════════════════════════════════════════════════
def discover_images(folder):
    """Return sorted image paths in a folder (png/jpg/jpeg), or [] if none."""
    if not os.path.isdir(folder):
        return []
    files = []
    for ext in ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG'):
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def draw_images(srand, files, n):
    """Pick n image paths without replacement where possible, else with
    replacement, using the cosmetic stream. Returns [] if the set is empty."""
    if not files:
        return []
    pool, out = list(files), []
    for _ in range(n):
        if not pool:
            pool = list(files)
        idx = int(srand() * len(pool))
        out.append(pool.pop(idx))
    return out


def build_bonus_deck(srand, n_food, n_neutral):
    """Shuffle a deck of food (True) and neutral (False) cues (cosmetic stream)."""
    deck = [True] * n_food + [False] * n_neutral
    for i in range(len(deck) - 1, 0, -1):
        j = int(srand() * (i + 1))
        deck[i], deck[j] = deck[j], deck[i]
    return deck


def build_bonus_positions(srand, n_bonus, n_trials, rev_trial):
    """Bandit-trial indices to insert bonuses after: spread across bins, buffered
    around the reversal, no two within BONUS_MIN_GAP. Returns a sorted list."""
    lo, hi, rev = CFG['BONUS_FIRST_AFTER'], n_trials - 1, rev_trial
    used = []

    def too_close(p):
        return (abs(p - rev) < CFG['BONUS_REV_BUFFER'] or
                any(abs(u - p) < CFG['BONUS_MIN_GAP'] for u in used))

    bin_w = (hi - lo) / n_bonus
    for i in range(n_bonus):
        a = int(lo + i * bin_w)
        b = int(lo + (i + 1) * bin_w)
        hi_i = max(a, b - 1)
        p, tries = a, 0
        while True:
            p = a + int(srand() * (hi_i - a + 1))
            tries += 1
            if not too_close(p) or tries >= 40:
                break
        used.append(p)
    return sorted(used)


# ════════════════════════════════════════════════════════════════════════════
#  Window, stimuli, and small drawing utilities
# ════════════════════════════════════════════════════════════════════════════
def build_window():
    """Open the experiment window in height units (resolution independent)."""
    win = visual.Window(size=WIN_SIZE, fullscr=FULLSCREEN, color=BG_COLOR,
                        units='height', allowGUI=False, winType='pyglet')
    win.mouseVisible = True
    return win


def make_sad_face(win, pos, size, ori=0.0):
    """A font-independent sad face (circle, two eyes, frown) as a stim list, so
    rendering does not depend on emoji-capable fonts."""
    r = size / 2.0
    head = visual.Circle(win, radius=r, pos=pos, fillColor=[1, 0.85, -0.6],
                        lineColor=[0.2, 0.0, -0.6], lineWidth=2, ori=ori)
    eye_dx, eye_dy, eye_r = r * 0.38, r * 0.30, r * 0.12
    eyes = [visual.Circle(win, radius=eye_r, pos=(pos[0] + sx * eye_dx, pos[1] + eye_dy),
                          fillColor=[-1, -1, -1], lineColor=[-1, -1, -1])
            for sx in (-1, 1)]
    # Frown: lower arc, drawn as a downward-curving polyline.
    pts = []
    for k in range(9):
        ang = math.radians(200 + 140 * k / 8.0)   # spans the lower mouth region
        pts.append((pos[0] + 0.45 * r * math.cos(ang),
                    pos[1] - 0.30 * r + 0.45 * r * math.sin(ang)))
    frown = visual.ShapeStim(win, vertices=pts, closeShape=False,
                            lineColor=[-1, -1, -1], lineWidth=4)
    return [head] + eyes + [frown]


def flip_clear(win, trig, photodiode_stim, pd_on):
    """Standard flip that also clears the trigger line and photodiode square."""
    if pd_on and photodiode_stim is not None:
        photodiode_stim.fillColor = [-1, -1, -1]
    t = win.flip()
    trig.clear()
    return t


def check_quit(win, trig, save_fn):
    """Abort-and-save on Escape (experimenter key). Returns nothing if no quit."""
    if 'escape' in event.getKeys(keyList=['escape']):
        save_fn()
        win.close()
        core.quit()


# ════════════════════════════════════════════════════════════════════════════
#  Incremental CSV writer (crash-safe: each row is flushed as it completes)
# ════════════════════════════════════════════════════════════════════════════
# A fixed superset of columns (bandit columns first, then bonus-only) keeps the
# schema identical to the web export while allowing per-row writing.
FIELDNAMES = [
    'participant_id', 'session', 'trial_type', 't_onset_s', 'seed',
    # bandit
    'trial', 'swap_count', 'position1', 'position2', 'position3',
    'p_reward_pos1', 'p_reward_pos2', 'p_reward_pos3',
    'choice', 'chosen_logo', 'rt_s', 'choice_late', 'outcome', 'points',
    'optimal_position', 'is_optimal', 'optimal_points', 'regret', 'cumulative_score',
    'cf_outcome_pos1', 'cf_outcome_pos2', 'cf_outcome_pos3',
    'cf_points_pos1', 'cf_points_pos2', 'cf_points_pos3',
    # bonus-only
    'bonus_trial_index', 'position_in_bandit_stream', 'food_set', 'cue_type',
    'food_bonus_cue', 'cue_image', 'cue_duration_ms', 'anticipatory_delay_ms',
    'adaptive_window_ms', 'target_response_key', 'target_rt_ms', 'premature_rt_ms',
    'target_hit', 'target_miss', 'target_too_fast', 'target_no_response',
    'bonus_points_earned', 'bonus_cumulative', 'bonus_hit_rate',
    'cue_onset_ms', 'delay_onset_ms', 'target_onset_ms', 'response_ms',
    'feedback_onset_ms', 'trigger_code',
]


class DataLog:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'w', newline='', encoding='utf-8')
        self.w = csv.DictWriter(self.f, fieldnames=FIELDNAMES,
                                restval='', extrasaction='ignore')
        self.w.writeheader()
        self.f.flush()

    def write(self, row):
        self.w.writerow(row)
        self.f.flush()          # survive a crash mid-session (important for patients)
        os.fsync(self.f.fileno())

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════════
#  Main experiment
# ════════════════════════════════════════════════════════════════════════════
def main():
    settings = run_dialog()
    seed = settings['seed']

    # Two independent streams from the one seed (main = schedule, srand = cosmetic).
    rand = make_rng(seed)
    srand = make_rng((seed ^ 0x9E3779B9) & 0xFFFFFFFF)

    # ---- Output file --------------------------------------------------------
    os.makedirs('data', exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    fname = 'sub-%s_ses-%s_task-bandit_%s.csv' % (settings['pid'], settings['session'], stamp)
    log = DataLog(os.path.join('data', fname))
    logging.LogFile(os.path.join('data', fname.replace('.csv', '.log')), level=logging.EXP)

    # ---- Window, triggers, photodiode --------------------------------------
    win = build_window()
    trig = Triggers(settings['trig_mode'], settings['parallel_addr'])
    pd_on = settings['photodiode']
    aspect = win.size[0] / float(win.size[1])
    pd_stim = visual.Rect(win, width=0.07, height=0.07,
                          pos=(aspect * 0.5 - 0.05, -0.5 + 0.05),
                          fillColor=[-1, -1, -1], lineColor=None) if pd_on else None

    # ---- Stimuli ------------------------------------------------------------
    sym_paths = {n: os.path.join(SYMBOL_DIR, n + '.png') for n in SYMBOL_NAMES}
    card_xs = [-0.42, 0.0, 0.42]
    cards = [visual.Rect(win, width=0.34, height=0.42, pos=(x, 0.0),
                         fillColor=[0.85, 0.85, 0.85], lineColor=[0.4, 0.4, 0.4],
                         lineWidth=2) for x in card_xs]
    sym_stims = [visual.ImageStim(win, size=0.24, pos=(x, 0.0)) for x in card_xs]

    header_score = visual.TextStim(win, text='', pos=(-aspect * 0.5 + 0.04, 0.45),
                                   height=0.035, color='white', anchorHoriz='left')
    header_trial = visual.TextStim(win, text='', pos=(aspect * 0.5 - 0.04, 0.45),
                                   height=0.035, color='white', anchorHoriz='right')
    prog_bg = visual.Rect(win, width=aspect, height=0.012, pos=(0, 0.49),
                          fillColor=[0.2, 0.2, 0.2], lineColor=None)
    prog_fg = visual.Rect(win, width=0.0001, height=0.012, pos=(0, 0.49),
                          fillColor=[0.1, 0.6, 0.4], lineColor=None, anchorHoriz='left')
    prompt = visual.TextStim(win, text='', pos=(0, -0.36), height=0.045, color='white')

    fb_center = visual.TextStim(win, text='', pos=(0, 0), height=0.10, bold=True)
    cue_stim = visual.ImageStim(win, size=0.5, pos=(0, 0))
    fix_stim = visual.TextStim(win, text='+', pos=(0, 0), height=0.08, color='white')
    square = visual.Rect(win, width=0.18, height=0.18, pos=(0, 0),
                         fillColor=[1, 1, 1], lineColor=None)
    big = visual.TextStim(win, text='', pos=(0, 0.05), height=0.07, color='white', bold=True)
    sub = visual.TextStim(win, text='', pos=(0, -0.08), height=0.04, color='white')
    kb = keyboard.Keyboard()

    # Corner geometry for bandit feedback images / sad faces.
    corner_pos = [(-aspect * 0.34, 0.30), (aspect * 0.34, 0.30),
                  (-aspect * 0.34, -0.30), (aspect * 0.34, -0.30)]
    corner_imgs = [visual.ImageStim(win, size=0.20, pos=p) for p in corner_pos]

    # ---- Image sets + session food set -------------------------------------
    images = {k: discover_images(v) for k, v in IMG_DIRS.items()}

    def food_set():
        """Assign sweet/savory once. 'auto' draws from the cosmetic stream; an
        explicit override is honoured without consuming a draw."""
        if state['food_set'] is None:
            if settings['food_override'] is not None:
                state['food_set'] = settings['food_override']
            else:
                state['food_set'] = 'sweet' if srand() < 0.5 else 'savory'
        return state['food_set']

    def win_files():
        return images['sweet'] if food_set() == 'sweet' else images['savory']

    # ---- Task state ---------------------------------------------------------
    state = dict(
        trial=0,                # 0-indexed completed bandit trials (mirrors web S.trial)
        score=0,
        profiles=[list(CFG['PROFILE_A']), list(CFG['PROFILE_B']), list(CFG['PROFILE_C'])],
        swap_count=0,
        food_set=None,
        bonus_count=0, bonus_score=0, bonus_hits=0,
        bonus_window=CFG['WIN_START'],
        task_start=None,
    )
    swap_idx = CFG['REVERSAL_TRIAL'] - 1   # 0-indexed reversal trial

    # Per-session placement (MAIN stream, in the web's order: symbolMap then slotOrder).
    symbol_map = shuffle3(rand)   # logical arm -> symbol index
    slot_order = shuffle3(rand)   # screen slot  -> logical arm

    def sym_name(arm):
        return SYMBOL_NAMES[symbol_map[arm]]

    def save_and_close():
        log.close()

    # ---- Timed-hold helper --------------------------------------------------
    def hold(drawables, dur_ms, trig_code=None):
        """Draw a static screen, flip (onset), pulse trigger + photodiode, hold."""
        for d in drawables:
            d.draw()
        if pd_on and trig_code is not None:
            pd_stim.fillColor = [1, 1, 1]
            pd_stim.draw()
        win.flip()                                  # onset
        if trig_code is not None:
            trig.send(trig_code)
        clk = core.Clock()
        while clk.getTime() < dur_ms / 1000.0:
            check_quit(win, trig, save_and_close)
            flip_clear(win, trig, pd_stim, pd_on)   # clears trigger + photodiode after 1 frame
            for d in drawables:
                d.draw()
            if pd_on:
                pd_stim.draw()
        return clk

    def update_header():
        header_score.text = 'Score: %d pts' % state['score']
        header_trial.text = 'Trial %d / %d' % (min(state['trial'] + 1, CFG['N_TRIALS']),
                                               CFG['N_TRIALS'])
        frac = state['trial'] / float(CFG['N_TRIALS'])
        prog_fg.width = max(0.0001, aspect * frac)
        prog_fg.pos = (-aspect * 0.5, 0.49)

    # ── Bandit trial ─────────────────────────────────────────────────────────
    def bandit_trial():
        z = state['trial']                          # 0-indexed index of THIS trial
        t_onset = core.getTime() - state['task_start']

        # Reversal check happens before sampling, exactly as in the web version.
        state['profiles'], swapped = apply_reversal(rand, state['profiles'], z, swap_idx)
        if swapped:
            state['swap_count'] += 1

        # Predetermined per-arm outcomes this trial (3 draws, arm order 0,1,2).
        cf_out = [sample_outcome(rand, p) for p in state['profiles']]
        cf_pts = [CFG['REWARD_PTS'] if o == 'reward' else CFG['LOSS_PTS'] for o in cf_out]

        # Draw choice screen; collect a mouse click (4 s -> nudge, then keep waiting).
        for c in cards:
            c.fillColor = [0.85, 0.85, 0.85]
        for i, slot in enumerate(slot_order):
            sym_stims[i].image = sym_paths[sym_name(slot)]
        prompt.text = 'Click a symbol to play'
        prompt.color = 'white'
        update_header()

        def draw_choice():
            prog_bg.draw(); prog_fg.draw(); header_score.draw(); header_trial.draw()
            for c in cards:
                c.draw()
            for s in sym_stims:
                s.draw()
            prompt.draw()

        draw_choice()
        if pd_on:
            pd_stim.fillColor = [1, 1, 1]; pd_stim.draw()
        win.flip()                                  # choice onset
        trig.send(EVENT_CODES['choice_onset'])
        clk = core.Clock()
        mouse = event.Mouse(win=win)
        mouse.clickReset()
        prev_down = False
        chosen_slot = None
        late = False
        while chosen_slot is None:
            check_quit(win, trig, save_and_close)
            flip_clear(win, trig, pd_stim, pd_on)
            draw_choice()
            if pd_on:
                pd_stim.draw()
            if not late and clk.getTime() >= CFG['CHOICE_DEADLINE_MS'] / 1000.0:
                late = True
                prompt.text = 'Please answer faster.'
                prompt.color = 'red'
            down = mouse.getPressed()[0]
            if down and not prev_down:
                for i, c in enumerate(cards):
                    if c.contains(mouse):
                        chosen_slot = i
                        rt = clk.getTime()
                        break
            prev_down = down

        arm = slot_order[chosen_slot]               # logical arm at the clicked slot
        trig.send(EVENT_CODES['choice_made'])

        # Brief "pull" animation: enlarge the chosen card, dim the others.
        anim = core.Clock()
        while anim.getTime() < CFG['ANIM_MS'] / 1000.0:
            check_quit(win, trig, save_and_close)
            flip_clear(win, trig, pd_stim, pd_on)
            prog_bg.draw(); prog_fg.draw(); header_score.draw(); header_trial.draw()
            for i, c in enumerate(cards):
                c.opacity = 1.0 if i == chosen_slot else 0.35
                c.draw()
            for i, s in enumerate(sym_stims):
                s.opacity = 1.0 if i == chosen_slot else 0.35
                s.draw()
        for c in cards:
            c.opacity = 1.0
        for s in sym_stims:
            s.opacity = 1.0

        # Resolve outcome and score (chosen arm's predetermined outcome).
        outcome = cf_out[arm]
        points = cf_pts[arm]
        state['score'] += points

        # Position-frame bookkeeping (1 = left .. 3 = right), matching the web export.
        def L(pos):
            return slot_order[pos - 1]
        choice_pos = slot_order.index(arm) + 1
        p_rewards = [pr[0] for pr in state['profiles']]
        opt_arm = p_rewards.index(max(p_rewards))
        optimal_pos = slot_order.index(opt_arm) + 1
        optimal_points = cf_pts[opt_arm]

        row = {
            'participant_id': settings['pid'], 'session': settings['session'],
            'trial_type': 'bandit', 't_onset_s': round(t_onset, 4), 'seed': seed,
            'trial': z + 1, 'swap_count': state['swap_count'],
            'position1': sym_name(L(1)), 'position2': sym_name(L(2)), 'position3': sym_name(L(3)),
            'p_reward_pos1': round(state['profiles'][L(1)][0], 4),
            'p_reward_pos2': round(state['profiles'][L(2)][0], 4),
            'p_reward_pos3': round(state['profiles'][L(3)][0], 4),
            'choice': choice_pos, 'chosen_logo': sym_name(arm),
            'rt_s': round(rt, 4), 'choice_late': 1 if late else 0,
            'outcome': outcome, 'points': points,
            'optimal_position': optimal_pos, 'is_optimal': 1 if choice_pos == optimal_pos else 0,
            'optimal_points': optimal_points, 'regret': optimal_points - points,
            'cumulative_score': state['score'],
            'cf_outcome_pos1': cf_out[L(1)], 'cf_outcome_pos2': cf_out[L(2)], 'cf_outcome_pos3': cf_out[L(3)],
            'cf_points_pos1': cf_pts[L(1)], 'cf_points_pos2': cf_pts[L(2)], 'cf_points_pos3': cf_pts[L(3)],
            'trigger_code': EVENT_CODES['bandit_win'] if outcome == 'reward' else EVENT_CODES['bandit_loss'],
        }
        log.write(row)
        state['trial'] += 1
        update_header()

        # Feedback: centered points + four tilted corner images (food) / sad faces (loss).
        is_win = outcome == 'reward'
        fb_center.text = ('+%d Points' % points) if points > 0 else ('%d Points' % points)
        fb_center.color = [0.1, 0.8, 0.3] if is_win else [0.9, 0.2, 0.2]
        tilts = [int(srand() * 27) - 13 for _ in range(4)]   # srandInt(-13,13)
        drawables = [fb_center]
        sad_stims = []
        if is_win:
            pics = draw_images(srand, win_files(), 4)
            for i, ci in enumerate(corner_imgs):
                ci.ori = tilts[i]
                if i < len(pics) and pics[i]:
                    ci.image = pics[i]
                    drawables.append(ci)
        else:
            for i, p in enumerate(corner_pos):
                sad_stims.extend(make_sad_face(win, p, 0.22, ori=tilts[i]))
            drawables.extend(sad_stims)

        hold(drawables, CFG['FEEDBACK_MS'],
             trig_code=(EVENT_CODES['bandit_win'] if is_win else EVENT_CODES['bandit_loss']))
        hold([], CFG['ITI_MS'])                     # blank ITI

    # ── Bonus (mini-MID) trial ───────────────────────────────────────────────
    def bonus_trial(is_food):
        state['bonus_count'] += 1
        t0 = core.getTime()
        t_onset = t0 - state['task_start']

        cue_type = ('food_%s' % food_set()) if is_food else 'neutral'
        files = win_files() if is_food else images['neutral']
        picked = draw_images(srand, files, 1)
        cue_path = picked[0] if picked else None
        delay_ms = CFG['DELAY_MIN_MS'] + int(srand() * (CFG['DELAY_MAX_MS'] - CFG['DELAY_MIN_MS'] + 1))
        window_ms = int(round(state['bonus_window']))

        # Phase onset timestamps (ms since t0), filled from real flip times.
        marks = dict(cue=0, delay=0, target=0, feedback=0)
        kb.clearEvents()
        kb.clock.reset()                            # key .rt measured from intro onset (t0)

        # 1) "Bonus round!" intro
        big.text = 'Bonus round!'; big.color = 'white'
        hold([big], CFG['BONUS_INTRO_MS'], trig_code=EVENT_CODES['bonus_intro'])
        kb.clearEvents()                            # ignore any press made during the intro

        press = {'key': '', 'rt_ms': None}

        def listen():
            """Return True if a key was caught this frame (records first press)."""
            keys = kb.getKeys(waitRelease=False, clear=True)
            if keys and press['rt_ms'] is None:
                press['key'] = keys[0].name
                press['rt_ms'] = keys[0].rt * 1000.0    # ms since t0 (kb clock)
                return True
            return False

        def phase(drawables, dur_ms, trig_code):
            """Hold a screen while listening; break early on the first key press."""
            for d in drawables:
                d.draw()
            if pd_on and trig_code is not None:
                pd_stim.fillColor = [1, 1, 1]; pd_stim.draw()
            on = win.flip()
            if trig_code is not None:
                trig.send(trig_code)
            clk = core.Clock()
            caught = False
            while clk.getTime() < dur_ms / 1000.0:
                check_quit(win, trig, save_and_close)
                flip_clear(win, trig, pd_stim, pd_on)
                for d in drawables:
                    d.draw()
                if pd_on:
                    pd_stim.draw()
                if press['rt_ms'] is None and listen():
                    caught = True
                    break
            return on, caught

        # 2) cue
        cue_trig = EVENT_CODES['cue_food'] if is_food else EVENT_CODES['cue_neutral']
        if cue_path:
            cue_stim.image = cue_path
            cue_draw = [cue_stim]
        else:
            sub.text = ('FOOD CUE' if is_food else 'NEUTRAL CUE') + '\n(placeholder)'
            sub.pos = (0, 0); sub.color = 'white'
            cue_draw = [sub]
        on, caught = phase(cue_draw, CFG['CUE_MS'], cue_trig)
        marks['cue'] = round((on - t0) * 1000.0)
        sub.pos = (0, -0.08)
        outcome = None
        if caught:                                   # press during cue -> premature
            outcome = 'too_fast'

        # 3) anticipatory fixation
        if outcome is None:
            on, caught = phase([fix_stim], delay_ms, EVENT_CODES['fixation'])
            marks['delay'] = round((on - t0) * 1000.0)
            if caught:
                outcome = 'too_fast'

        # 4) target square (response window)
        if outcome is None:
            on, caught = phase([square], window_ms, EVENT_CODES['target'])
            marks['target'] = round((on - t0) * 1000.0)
            if caught:
                outcome = 'hit'
                trig.send(EVENT_CODES['response'])

        # 5) grace window (late press still logged as miss-with-RT)
        if outcome is None:
            on, caught = phase([], CFG['GRACE_MS'], None)
            if caught:
                outcome = 'miss'
            else:
                outcome = 'no_response'

        # Classify RT fields relative to the right phase onset.
        prem_rt = rt_ms = ''
        resp_ms = ''
        if press['rt_ms'] is not None:
            resp_ms = round(press['rt_ms'])
            if outcome == 'too_fast':
                prem_rt = round(press['rt_ms'] - marks['cue'])
            else:
                rt_ms = round(press['rt_ms'] - marks['target'])

        # Staircase + tallies (too_fast leaves the window unchanged).
        hit = outcome == 'hit'
        if hit:
            state['bonus_window'] = max(CFG['WIN_FLOOR'], state['bonus_window'] - CFG['WIN_STEP_DOWN'])
        elif outcome != 'too_fast':
            state['bonus_window'] = min(CFG['WIN_CEIL'], state['bonus_window'] + CFG['WIN_STEP_UP'])
        pts = CFG['BONUS_PTS'] if hit else 0
        state['bonus_score'] += pts
        state['bonus_hits'] += 1 if hit else 0

        # 6) feedback (no pictures)
        if hit:
            big.text = 'Congrats!'; big.color = [0.1, 0.8, 0.3]
            sub.text = '+%d points' % CFG['BONUS_PTS']
        elif outcome == 'too_fast':
            big.text = 'Too soon!'; big.color = [0.95, 0.6, 0.1]
            sub.text = 'Wait for the square.'
        else:
            big.text = 'Next time respond faster.'; big.color = [0.95, 0.6, 0.1]
            sub.text = ''
        fb_on = core.getTime()
        fb_clk = hold([big, sub] if sub.text else [big],
                      CFG['BONUS_FEEDBACK_MS'], trig_code=EVENT_CODES['bonus_feedback'])
        marks['feedback'] = round((fb_on - t0) * 1000.0)

        row = {
            'participant_id': settings['pid'], 'session': settings['session'],
            'trial_type': 'bonus_food' if is_food else 'bonus_neutral',
            't_onset_s': round(t_onset, 4), 'seed': seed,
            'bonus_trial_index': state['bonus_count'],
            'position_in_bandit_stream': state['trial'],
            'food_set': food_set(), 'cue_type': cue_type,
            'food_bonus_cue': 1 if is_food else 0,
            'cue_image': cue_path or '',
            'cue_duration_ms': CFG['CUE_MS'], 'anticipatory_delay_ms': delay_ms,
            'adaptive_window_ms': window_ms,
            'target_response_key': press['key'], 'target_rt_ms': rt_ms, 'premature_rt_ms': prem_rt,
            'target_hit': 1 if hit else 0,
            'target_miss': 1 if outcome == 'miss' else 0,
            'target_too_fast': 1 if outcome == 'too_fast' else 0,
            'target_no_response': 1 if outcome == 'no_response' else 0,
            'bonus_points_earned': pts, 'bonus_cumulative': state['bonus_score'],
            'bonus_hit_rate': round(state['bonus_hits'] / state['bonus_count'], 3),
            'cue_onset_ms': marks['cue'], 'delay_onset_ms': marks['delay'],
            'target_onset_ms': marks['target'], 'response_ms': resp_ms,
            'feedback_onset_ms': marks['feedback'],
            'trigger_code': EVENT_CODES['response'] if hit else EVENT_CODES['bonus_feedback'],
        }
        log.write(row)

    # ---- Instructions -------------------------------------------------------
    instr = visual.TextStim(
        win, color='white', height=0.034, wrapWidth=1.2, pos=(0, 0.05),
        text=("Choose one of three symbols on each trial by clicking it.\n"
              "You will win 10 points or lose 10 points. Some symbols win more "
              "often than others, and the best one can change during the task, so "
              "keep paying attention and use what you learn.\n\n"
              "Every so often a short Bonus round appears: you will see a picture, "
              "then a cross, then a white square. The moment the square appears, "
              "press ANY KEY as fast as you can to earn bonus points. Wait for the "
              "square; pressing too early does not count.\n\n"
              "The whole task takes about 20 minutes.\n\n"
              "Press SPACE to begin."))
    instr.draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    win.mouseVisible = True

    # ---- Bonus schedule (cosmetic stream, in the web's order) ---------------
    food_set()                                       # fix the session food set first
    deck = build_bonus_deck(srand, CFG['N_BONUS_FOOD'], CFG['N_BONUS_NEUTRAL'])
    positions = build_bonus_positions(srand, len(deck), CFG['N_TRIALS'], CFG['REVERSAL_TRIAL'])
    insert_at = {p: deck[i] for i, p in enumerate(positions)}

    # ---- Run: 200 bandit trials with bonuses interleaved --------------------
    state['task_start'] = core.getTime()
    for t in range(1, CFG['N_TRIALS'] + 1):
        bandit_trial()
        if t in insert_at:
            bonus_trial(insert_at[t])

    # ---- End screen ---------------------------------------------------------
    dur_min = (core.getTime() - state['task_start']) / 60.0
    end = visual.TextStim(
        win, color='white', height=0.04, wrapWidth=1.2,
        text=("Great work, you finished the task.\n\n"
              "Final score: %d pts\n"
              "Bonus hits: %d / %d\n"
              "Total time: %.1f min\n\n"
              "Your responses have been saved.\nPress SPACE to exit."
              % (state['score'], state['bonus_hits'], state['bonus_count'], dur_min)))
    end.draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    save_and_close()
    win.close()
    core.quit()


if __name__ == '__main__':
    main()
