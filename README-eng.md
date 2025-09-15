# PID-GA: Streaming PID Tuning with Genetic Algorithm + Auto-Load Last Run

A **Gradio** app to **tune PID** controllers using a **Genetic Algorithm (GA)** with **live/streaming** updates.  
Each generation:

- 📈 Shows the **step response** of the **best-fitness** individual
- 🟢 Logs **one image per generation**: the **minimum-overshoot** candidate (`NNN_minOS.png`)
- 📋 Appends **one table row per generation** (Kp, Ki, Kd, Overshoot, Fitness, etc.)
- 💾 Auto-saves `rows.json` and a CSV so next sessions can **auto-load** the last run

> ✨ On app start, if there’s any `pid_ga_runs/run_*` folder, the app **automatically loads** the **latest run** (table + gallery) — **no need to click Run**.

---

## Features

- **Live/Streaming GA**: updates **every generation**
- **Min-Overshoot Logger**: exactly **1 image per generation** (tie-break by fitness)
- **Smart Early-Stop (optional)**:
  - **Patience** + **Min Δfitness**
  - Stop when **Fitness ≤ target**
  - Stop when **Overshoot ≤ target**
- **Auto-Load Last Run** at startup (from `rows.json` / CSV / images)
- **Local-only cache & logs** (safe on systems with `/tmp` restrictions)
  - `.gradio_tmp/` (Gradio cache)
  - `pid_ga_runs/run_<timestamp>_<id>/` (images + logs)
- **Export CSV** (single click)

---

## Quick Start

```bash
pip install --upgrade gradio matplotlib numpy
python pid_ga_streaming_earlystop.py
# open http://127.0.0.1:7860
