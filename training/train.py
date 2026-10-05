"""
Train the 784-32-16-10 MNIST network used in the story, and export everything
the website needs:

  public/data/network.bin   float32 weights/biases (trained, then untrained)
  public/data/network.json  layout of network.bin, the "hero" 7 image, and the
                            real numbers quoted in the narration

Plain NumPy, no deep-learning framework. Deterministic (fixed seed).

Usage:
  pip install numpy
  sh training/fetch_mnist.sh   # or download the four MNIST .gz files into training/data/
  python training/train.py
"""

import gzip
import json
import os
import struct

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "..", "public", "data")

SIZES = [784, 32, 16, 10]
EPOCHS = 15
BATCH = 64
LR = 1e-3
SEED = 7

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------- data

def load_images(name):
    with gzip.open(os.path.join(DATA, name), "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), np.uint8).reshape(n, rows * cols)


def load_labels(name):
    with gzip.open(os.path.join(DATA, name), "rb") as f:
        f.read(8)
        return np.frombuffer(f.read(), np.uint8)


x_train = load_images("train-images-idx3-ubyte.gz").astype(np.float32) / 255.0
y_train = load_labels("train-labels-idx1-ubyte.gz")
x_test_u8 = load_images("t10k-images-idx3-ubyte.gz")
x_test = x_test_u8.astype(np.float32) / 255.0
y_test = load_labels("t10k-labels-idx1-ubyte.gz")


# ---------------------------------------------------------------- model

def init_params():
    """He initialisation for ReLU layers; biases start at zero."""
    params = []
    for n_in, n_out in zip(SIZES[:-1], SIZES[1:]):
        w = rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out)).astype(np.float32)
        b = np.zeros(n_out, np.float32)
        params.append([w, b])
    return params


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params, x):
    """Returns the list of activations [input, h1, h2, probs] and the logits."""
    acts = [x]
    h = x
    for i, (w, b) in enumerate(params):
        z = h @ w + b
        if i < len(params) - 1:
            h = np.maximum(z, 0.0)  # ReLU
        else:
            logits = z
            h = softmax(z)
        acts.append(h)
    return acts, logits


def accuracy(params, x, y):
    probs = forward(params, x)[0][-1]
    return float((probs.argmax(axis=1) == y).mean())


def copy_params(params):
    return [[w.copy(), b.copy()] for w, b in params]


# ---------------------------------------------------------------- training

params = init_params()
untrained = copy_params(params)

# Adam optimiser state
m = [[np.zeros_like(w), np.zeros_like(b)] for w, b in params]
v = [[np.zeros_like(w), np.zeros_like(b)] for w, b in params]
beta1, beta2, eps = 0.9, 0.999, 1e-8

# Snapshots so we can later show the hero 7's confidence climbing during training.
SNAPSHOT_STEPS = {0, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400}
snapshots = {0: copy_params(params)}

step = 0
n = len(x_train)
for epoch in range(EPOCHS):
    order = rng.permutation(n)
    for start in range(0, n, BATCH):
        idx = order[start:start + BATCH]
        xb, yb = x_train[idx], y_train[idx]
        acts, _ = forward(params, xb)

        # Cross-entropy gradient w.r.t. logits: probs - one_hot
        delta = acts[-1].copy()
        delta[np.arange(len(yb)), yb] -= 1.0
        delta /= len(yb)

        # Backpropagation, last layer first (chain rule, layer by layer)
        grads = [None] * len(params)
        for i in reversed(range(len(params))):
            w, _ = params[i]
            grads[i] = [acts[i].T @ delta, delta.sum(axis=0)]
            if i > 0:
                delta = (delta @ w.T) * (acts[i] > 0)  # ReLU derivative

        step += 1
        for i in range(len(params)):
            for j in range(2):
                g = grads[i][j]
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * g
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * g * g
                mh = m[i][j] / (1 - beta1 ** step)
                vh = v[i][j] / (1 - beta2 ** step)
                params[i][j] -= LR * mh / (np.sqrt(vh) + eps)

        if step in SNAPSHOT_STEPS:
            snapshots[step] = copy_params(params)

    print(f"epoch {epoch + 1:2d}  train acc {accuracy(params, x_train, y_train):.4f}"
          f"  test acc {accuracy(params, x_test, y_test):.4f}")

snapshots[step] = copy_params(params)
test_acc = accuracy(params, x_test, y_test)

# ---------------------------------------------------------------- pick the hero 7
# A correctly classified test-set 7 that looks like a plain textbook seven:
# confident but not saturated, with "1" as the runner-up (as the story says).

probs_test = forward(params, x_test)[0][-1]
mean_seven = x_train[y_train == 7].mean(axis=0)
mean_seven /= np.linalg.norm(mean_seven)
candidates = []
candidates_all = []
for i in np.where(y_test == 7)[0]:
    p = probs_test[i]
    ranked = np.argsort(-p)
    typ = float(np.dot(x_test[i], mean_seven) / (np.linalg.norm(x_test[i]) + 1e-6))
    if ranked[0] == 7:
        candidates_all.append((typ, i))
    if ranked[0] == 7 and ranked[1] == 1 and 0.95 <= p[7] <= 0.995:
        # Favour the most "typical" seven: closest to the average training 7.
        candidates.append((typ, i))
if not candidates:
    raise SystemExit("No hero candidate found; loosen the filter.")
hero = max(candidates)[1]

hero_x = x_test[hero:hero + 1]
hero_acts, hero_logits = forward(params, hero_x)
untrained_acts, _ = forward(untrained, hero_x)

def r(a, d=4):
    return [round(float(x), d) for x in np.asarray(a).ravel()]


curve = []
for s in sorted(snapshots):
    p = forward(snapshots[s], hero_x)[0][-1][0]
    curve.append({"step": int(s), "probs": r(p)})

# Three other real sevens written in different styles, for the "ghost sevens" in beat 1.5.
# Taken from the less typical end of the correctly classified test sevens.
ghost_pool = sorted(c for c in candidates_all if c[1] != hero)
ghosts = [ghost_pool[int(q * (len(ghost_pool) - 1))][1] for q in (0.02, 0.08, 0.2)]

# A random "scribble" input: a few thick random strokes, for beat 6.9.
scribble = np.zeros((28, 28), np.float32)
srng = np.random.default_rng(3)
for _ in range(4):
    x0, y0, x1, y1 = srng.uniform(4, 24, 4)
    for t in np.linspace(0, 1, 60):
        cx, cy = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        yy, xx = np.mgrid[0:28, 0:28]
        scribble = np.maximum(scribble, np.clip(1.3 - np.hypot(xx - cx, yy - cy), 0, 1))
scribble_probs = forward(params, scribble.reshape(1, 784))[0][-1][0]

# ---------------------------------------------------------------- export

os.makedirs(OUT, exist_ok=True)


def flat(ps):
    return [a.astype(np.float32).ravel() for w, b in ps for a in (w, b)]


trained_arrays = flat(params)
untrained_arrays = flat(untrained)

layout = []
offset = 0
for label, arrays in (("trained", trained_arrays), ("untrained", untrained_arrays)):
    for li, (w, b) in enumerate(zip(arrays[0::2], arrays[1::2])):
        layout.append({"set": label, "layer": li, "kind": "weights",
                       "offset": offset, "length": int(w.size),
                       "shape": [SIZES[li], SIZES[li + 1]]})
        offset += w.size
        layout.append({"set": label, "layer": li, "kind": "biases",
                       "offset": offset, "length": int(b.size),
                       "shape": [SIZES[li + 1]]})
        offset += b.size

np.concatenate(trained_arrays + untrained_arrays).astype("<f4").tofile(
    os.path.join(OUT, "network.bin"))


meta = {
    "sizes": SIZES,
    "activation": "relu",
    "output": "softmax",
    "paramCount": int(sum(a.size for a in trained_arrays)),
    "weightsLayout": layout,
    "training": {
        "dataset": "MNIST", "trainImages": int(len(x_train)), "testImages": int(len(x_test)),
        "epochs": EPOCHS, "batchSize": BATCH, "optimizer": "Adam", "learningRate": LR,
        "seed": SEED, "steps": int(step),
    },
    "testAccuracy": round(test_acc, 4),
    "hero": {
        "testIndex": int(hero),
        "label": 7,
        "pixels": [int(p) for p in x_test_u8[hero]],  # 0-255, row-major 28x28
        "trained": {
            "hidden1": r(hero_acts[1]),
            "hidden2": r(hero_acts[2]),
            "probs": r(hero_acts[3]),
            "logits": r(hero_logits),
        },
        "untrained": {
            "hidden1": r(untrained_acts[1]),
            "hidden2": r(untrained_acts[2]),
            "probs": r(untrained_acts[3]),
        },
        "trainingCurve": curve,
    },
    "ghostSevens": [[int(p) for p in x_test_u8[g]] for g in ghosts],
    "scribble": {
        "pixels": [int(round(p * 255)) for p in scribble.ravel()],
        "probs": r(scribble_probs),
    },
}
with open(os.path.join(OUT, "network.json"), "w") as f:
    json.dump(meta, f, separators=(",", ":"))

print("\nhero test index:", hero)
print("test accuracy:", test_acc)
print("trained probs:  ", r(hero_acts[3], 4))
print("untrained probs:", r(untrained_acts[3], 4))
print("curve p7:", [(c["step"], c["probs"][7]) for c in curve])
print("scribble probs: ", r(scribble_probs, 3))
print("hidden1 active:", int((hero_acts[1] > 0).sum()), "/ 32;",
      "hidden2 active:", int((hero_acts[2] > 0).sum()), "/ 16")

# ---------------------------------------------------------------- time-lapse data (beats 7.12-7.14)
# Loaded lazily by the site after it starts, so it never delays the first frame.
#   timelapse.bin   int8 weights for each intermediate training snapshot, then
#                   uint8 training digits, then uint8 test digits
#   timelapse.json  offsets, quantisation scales, biases, labels and predictions

steps_sorted = sorted(snapshots)
blobs = []
offset = 0
snap_meta = []
for k, s in enumerate(steps_sorted):
    entry = {"step": int(s), "layers": []}
    for li, (w, b) in enumerate(snapshots[s]):
        info = {
            "p995": float(np.quantile(np.abs(w), 0.995)),  # display normalisation, as in the site
            "biases": r(b, 5),
        }
        if 0 < k < len(steps_sorted) - 1:  # endpoints are already in network.bin as float32
            scale = float(np.abs(w).max()) or 1.0
            q = np.clip(np.round(w / scale * 127), -127, 127).astype(np.int8)
            info.update({"offset": offset, "length": int(q.size), "scale": scale / 127})
            blobs.append(q.tobytes())
            offset += q.size
        entry["layers"].append(info)
    snap_meta.append(entry)

trng = np.random.default_rng(5)
train_pick = trng.choice(len(x_train), 192, replace=False)
train_u8 = (x_train[train_pick] * 255).round().astype(np.uint8)
blobs.append(train_u8.tobytes())
train_offset = offset
offset += train_u8.size

# 52 unseen test digits: 50 correct + 2 misses = 96.2%, matching the test accuracy.
pred_test = probs_test.argmax(axis=1)
wrong = np.where(pred_test != y_test)[0]
right = np.where(pred_test == y_test)[0]
test_pick = np.concatenate([trng.choice(right, 50, replace=False), trng.choice(wrong, 2, replace=False)])
# One miss in each half (the first misses no earlier than digit 6), so both the full run
# (50/52) and the shorter reduced-motion run of the first 26 (25/26) show 96.2% accuracy.
def misses_ok(order):
    m = np.where(pred_test[order] != y_test[order])[0]
    return 6 <= m[0] < 26 <= m[1]
trng.shuffle(test_pick)
while not misses_ok(test_pick):
    trng.shuffle(test_pick)
test_u8 = x_test_u8[test_pick]
blobs.append(test_u8.tobytes())
test_offset = offset
offset += test_u8.size

with open(os.path.join(OUT, "timelapse.bin"), "wb") as f:
    for blob in blobs:
        f.write(blob)

timelapse = {
    "snapshots": snap_meta,
    "trainDigits": {"offset": train_offset, "count": int(len(train_pick)),
                    "labels": [int(v) for v in y_train[train_pick]]},
    "testDigits": {
        "offset": test_offset, "count": int(len(test_pick)),
        "indices": [int(v) for v in test_pick],
        "labels": [int(v) for v in y_test[test_pick]],
        "preds": [int(v) for v in pred_test[test_pick]],
        "probs": [r(probs_test[i], 3) for i in test_pick],
    },
}
with open(os.path.join(OUT, "timelapse.json"), "w") as f:
    json.dump(timelapse, f, separators=(",", ":"))

print("timelapse.bin bytes:", offset, "| test digits correct:",
      int((pred_test[test_pick] == y_test[test_pick]).sum()), "/", len(test_pick))
