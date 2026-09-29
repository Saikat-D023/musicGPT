# microgpt-for-music

A tiny GPT that learns to compose melodies. Pure Python, zero dependencies, one file.

Same algorithm as @karpathy's microgpt, different domain:
instead of learning the pattern of names, it learns the pattern of music. Based on @kaizenman's micromusic.

## How it works

Each melody is one line of text. A note is two tokens, pitch then duration, and `BAR` marks a measure boundary:

```
C3 q C3 q D3 q E3 q BAR G3 h
```

Durations: `w` whole, `h` half, `q` quarter, `e` eighth, `s` sixteenth.

`train.py` contains everything: a scalar autograd engine (`Value`), a 1-layer transformer
(4 heads, 16-dim embeddings, about 6k parameters), the Adam optimizer, a sampler, and a small MIDI writer.

## Run

Requires Python 3.8+. No packages to install.

```bash
python train.py          # train for 1000 steps, then print 20 generated melodies
python train.py --save   # same, and also write melody_01.mid ... melody_20.mid
```

On the first run it downloads the melody dataset (about 11.5k melodies) to `input.txt`.
Training takes around 10 minutes on a single CPU core. Open the `.mid` files in any MIDI player to listen.

Example output after training:

```
sample  5: G3 e B3 e B3 e C4 e BAR D4 e D4 e D4 e D4 e
sample 11: D4 q F#4 q G4 q G4 q BAR B4 q C5 q A#4 q
```

## Test

```bash
python test.py
```

Checks the autograd engine's forward and backward pass on a small expression. Runs instantly.
