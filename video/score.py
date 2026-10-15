"""Procedural score for the CardioMap film (60 s, 48 kHz stereo): heartbeat synced to the ECG trace,
evolving drone (A minor -> A major at the outro), risers into each cut, impacts and pings on key beats.
Original and rights-free. Writes video/assets/score.wav."""
import wave
from pathlib import Path

import numpy as np

SR, DUR = 48000, 60.0
N = int(SR * DUR)
t = np.arange(N) / SR
L = np.zeros(N); R = np.zeros(N)
rng = np.random.default_rng(7)
BEAT = 60 / 72
CUTS = [5.0, 11.2, 24.0, 34.0, 43.4, 52.4]


def add(sig, start, gain=1.0, pan=0.0):
    i = int(start * SR); j = min(N, i + len(sig))
    if j <= i: return
    L[i:j] += sig[: j - i] * gain * (1 - max(0, pan)); R[i:j] += sig[: j - i] * gain * (1 + min(0, pan))


def env(n, a, d):
    x = np.arange(n) / SR
    return np.minimum(1, x / max(a, 1e-4)) * np.exp(-x / d)


def thump(f0=58, dur=.32, drop=.55):
    n = int(SR * dur); x = np.arange(n) / SR
    f = f0 * (1 + drop * np.exp(-x / .03))
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * env(n, .003, .09)
    click = rng.standard_normal(n) * env(n, .0005, .006) * .25
    return np.tanh((body + click) * 1.6)


def lowpass(x, cut):
    a = np.exp(-2 * np.pi * cut / SR); y = np.zeros_like(x); acc = 0.0
    for i in range(len(x)):
        acc = (1 - a) * x[i] + a * acc; y[i] = acc
    return y


# --- heartbeat: lub on the ECG QRS (phase .38), dub ~0.28 s later; loud in the cold open, then under
def hb_gain(tt):
    if tt < 5.0: return 1.0
    if tt > 58.6: return 1.0
    if 52.4 < tt < 58.6: return .55
    return .28
k = 0
while True:
    tl = k * BEAT + .38 * BEAT
    if tl > 59.6: break
    g = hb_gain(tl)
    add(thump(58), tl, .9 * g); add(thump(70, .26, .4), tl + .28, .55 * g)
    k += 1

# --- drone pad (additive saws through a slow lowpass), chords per section
def pad(freqs, start, end, gain, bright):
    n = int((end - start) * SR); x = np.arange(n) / SR
    sig = np.zeros(n)
    for f in freqs:
        for h in range(1, 7):
            det = 1 + (rng.random() - .5) * .004
            sig += np.sin(2 * np.pi * f * h * det * x + rng.random() * 6.28) / (h ** (2.2 - bright))
    fade = np.minimum(1, x / 1.5) * np.minimum(1, (x[-1] - x) / 1.5)
    trem = 1 + .08 * np.sin(2 * np.pi * .25 * x)
    s = sig * fade * trem
    add(s / np.abs(s).max(), start, gain, -.15); add(s / np.abs(s).max(), start + .012, gain * .9, .15)
Am = [110, 164.81, 261.63, 246.94]; F = [87.31, 130.81, 220, 261.63]; Dm = [73.42, 146.83, 220, 174.61]; A = [110, 164.81, 277.18, 329.63]
pad(Am, 0.5, 11.6, .10, .2)
pad(F, 11.0, 24.4, .13, .6)
pad(Dm, 23.8, 34.4, .12, .3)
pad(Am, 33.8, 43.8, .12, .5)
pad(F, 43.2, 52.8, .13, .7)
pad(A, 52.2, 60.0, .15, .8)

# --- pulse bass (eighths) from the heart reveal on
for kk in range(int((52.4 - 11.2) / (BEAT / 2))):
    tt = 11.2 + kk * BEAT / 2
    if 24.0 <= tt < 25.0 or 34.0 <= tt < 34.6: continue
    n = int(.18 * SR); x = np.arange(n) / SR
    root = {0: 55.0, 1: 43.65, 2: 36.71, 3: 55.0, 4: 43.65}[min(4, sum(tt >= c for c in CUTS[1:]) - 1)]
    sig = np.tanh(np.sin(2 * np.pi * root * x) * 2) * env(n, .004, .07)
    add(sig, tt, .16)

# --- risers + impacts
for c in CUTS:
    n = int(1.3 * SR); x = np.arange(n) / SR
    noise = rng.standard_normal(n)
    sweep = lowpass(noise, 1200) * (x / x[-1]) ** 2.5
    add(sweep / np.abs(sweep).max(), c - 1.3, .18, -.3); add(sweep / np.abs(sweep).max(), c - 1.29, .18, .3)
    add(thump(42, .9, .9), c, .75)
    burst = lowpass(rng.standard_normal(int(.5 * SR)), 3000) * env(int(.5 * SR), .001, .12)
    add(burst / np.abs(burst).max(), c, .22)

# --- pings on number landings
def ping(f, tt, g=.12, pan=0):
    n = int(1.2 * SR); x = np.arange(n) / SR
    s = (np.sin(2 * np.pi * f * x) + .3 * np.sin(2 * np.pi * f * 2.01 * x)) * env(n, .002, .35)
    add(s, tt, g, pan); add(s * .5, tt + .19, g * .5, -pan)  # echo
for i, tt in enumerate([14.0, 14.9, 15.8]): ping([659.25, 587.33, 523.25][i], tt, .10, [.4, .4, -.4][i])
ping(880, 18.2, .12)
for i, tt in enumerate([27.0, 27.3, 27.6]): ping(440, tt, .06)
for i, tt in enumerate([31.65, 31.9, 32.15]): ping([392, 349.23, 349.23][i], tt, .12)
for i in range(8): ping(1046.5, 35.1 + i * .14, .035, (-1) ** i * .3)
for i, tt in enumerate([45.0, 45.25, 45.5, 45.75]): ping([523.25, 659.25, 783.99, 1046.5][i], tt, .10)
ping(1318.5, 48.4, .1)
for i, tt in enumerate([54.0, 54.3, 54.6]): ping([440, 554.37, 659.25][i], tt, .09)

# --- whoosh for chips flying in
n = int(1.6 * SR); x = np.arange(n) / SR
wh = lowpass(rng.standard_normal(n), 2500) * np.sin(np.pi * x / x[-1]) ** 2
add(wh / np.abs(wh).max(), 5.0, .2, -.4); add(wh / np.abs(wh).max(), 5.05, .2, .4)
n = int(1.4 * SR); x = np.arange(n) / SR
wh2 = lowpass(rng.standard_normal(n), 1800) * (x / x[-1]) ** 2
add(wh2 / np.abs(wh2).max(), 9.6, .2)

# --- master: fade in/out, soft clip, normalise
fade = np.minimum(1, t / .4) * np.minimum(1, (DUR - t) / .8)
mx = np.stack([L, R], 1) * fade[:, None]
mx = np.tanh(mx * 1.3) / np.tanh(1.3)
mx = mx / np.abs(mx).max() * .89
out = Path(__file__).resolve().parent / "assets" / "score.wav"
with wave.open(str(out), "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mx * 32767).astype(np.int16).tobytes())
print("wrote", out)
