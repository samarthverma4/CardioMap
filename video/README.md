# CardioMap film (60 s)

A motion-graphics trailer rendered from the real product: the actual heart model (`frontend/public/heart.glb`),
real predictions for sample patient C and real metrics (`assets/data.json`, exported from the running API).

- `film.html`: frame-exact composition (three.js + SVG + DOM), driven by `window.renderFrame(t)`
- `render.py`: renders stills or all frames with headless Brave/Chrome on the GPU
- `score.py`: procedural, rights-free score (heartbeat synced to the ECG trace, drone, risers, impacts)

```bash
python -m http.server 8123 --bind 127.0.0.1            # from the repo root
python video/score.py                                   # -> video/assets/score.wav
python video/render.py frames /tmp/frames 30 0 1800     # 1800 JPEG frames
ffmpeg -framerate 30 -i /tmp/frames/f%05d.jpg -i video/assets/score.wav -c:v libx264 -crf 17 -pix_fmt yuv420p \
  -af "loudnorm=I=-14:TP=-1.5:LRA=11" -c:a aac -b:a 256k -shortest video/CardioMap_film.mp4
```
