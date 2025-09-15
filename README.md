# PID-GA — Streaming PID Tuning dengan Genetic Algorithm + Auto-Load Run Terakhir

Aplikasi **Gradio** untuk _tuning_ **PID** memakai **Genetic Algorithm (GA)** secara **live/streaming**.

Setiap generasi:
- Menampilkan **grafik respons step** (individu **best fitness**),
- Menyimpan & menampilkan **gambar kandidat dengan _overshoot_ terkecil**,
- Mengisi **tabel ringkasan** (1 baris per generasi),
- **Auto-save** `rows.json` & `CSV` → sesi berikutnya **auto-load** tanpa harus klik **Run**.

> ✨ Saat aplikasi dibuka, jika ada folder `pid_ga_runs/run_*` dari eksekusi sebelumnya, **tabel & galeri langsung terisi** dari **run terbaru**.

---

## 🔎 Fitur Utama

- **Streaming**: plot **best-of-generation** diperbarui **setiap generasi**.
- **Log _min-overshoot_**: per generasi menyimpan **1 gambar** `NNN_minOS.png` (tie-break pakai fitness).
- **Early Stop cerdas (opsional)**:
  - **Patience** + **Min ΔFitness** (berhenti jika tak ada perbaikan),
  - Target **Fitness ≤ X**,
  - Target **Overshoot ≤ Y%**.
- **Auto-Load last run** saat startup (dari `rows.json` / `CSV` / fallback gambar).
- **Folder lokal** (di sebelah script):
  - `.gradio_tmp/` (cache Gradio)
  - `pid_ga_runs/run_<timestamp>_<id>/` (gambar & log)
- **Export CSV** dari tabel (sekali klik).

---
## 🗂️ Struktur Folder
```bash
├─ pid_ga_streaming_earlystop.py
├─ .gradio_tmp/                   # cache Gradio (otomatis)
└─ pid_ga_runs/
  └─ run_2025..._<id>/
      ├─ 000_minOS.png
      ├─ 001_minOS.png
      ├─ ...
      ├─ rows.json               # tabel ringkasan (auto-save tiap generasi)
      └─ log_min_overshoot.csv   # auto-save saat berhenti
```
---
## 🚀 Demo Cepat

```bash
pip install --upgrade gradio matplotlib numpy
python pid_ga_streaming_earlystop.py
# buka http://127.0.0.1:7860
