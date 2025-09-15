## PID-GA — Simulasi PID Tuning dengan Genetic Algorithm 

<img width="2035" height="1294" alt="Screenshot 2025-09-15 123418" src="https://github.com/user-attachments/assets/cbacafef-fc64-4d66-afa9-8e51c3a7655a" />
<img width="1892" height="775" alt="Screenshot 2025-09-15 123620" src="https://github.com/user-attachments/assets/35dec3a8-791f-4bf2-931b-1f193c63b034" />

## Deskripsi Singkat: PID & Algoritma Genetika

### PID (Proportional–Integral–Derivative)
**PID controller** adalah pengendali umpan-balik yang menghitung sinyal kendali dari kombinasi tiga aksi: proporsional (P), integral (I), dan derivatif (D) terhadap error (e(t)=r(t)-y(t)\).

<img width="370" height="72" alt="Screenshot 2025-09-15 124420" src="https://github.com/user-attachments/assets/d30ae1db-cfa1-4e7a-8719-39de47a81884" />

- **P (Proportional)**: merespons sebanding dengan error saat ini → mempercepat menuju setpoint; terlalu besar dapat memicu **overshoot**/osilasi.  
- **I (Integral)**: mengakumulasi error sebelumnya → **menghilangkan steady-state error**; berlebihan dapat memicu **windup** dan overshoot.  
- **D (Derivative)**: mengantisipasi perubahan error → **meredam overshoot** dan meningkatkan stabilitas; sensitif terhadap **noise**.

**Metrik kinerja** umum: **Overshoot (%)**, **Settling time (Ts)**, **Rise time (Tr)**, **IAE/ISE** (integral error), dan **energi aktuasi**.

> Implementasi digital (diskrit) memakai penjumlahan kontribusi \(K_p e[k]\), integral diskrit (akumulator), dan beda hingga untuk turunan.

---

### Algoritma Genetika (Genetic Algorithm, GA)
**Algoritma Genetika** adalah metode optimasi berbasis evolusi yang mencari solusi melalui:

1. **Inisialisasi populasi** kandidat random/acak.  
2. **Evaluasi fitness** tiap kandidat disesuaikan dg fungsi tujuan.  
3. **Seleksi** kandidat unggul → **crossover** (rekombinasi) → **mutasi** kecil menjaga keragaman.  
4. **Elitisme**: menyalin kandidat terbaik ke generasi berikutnya.  
5. **Terminasi** saat mencapai jumlah generasi, **patience** (tidak ada perbaikan berarti), atau target performa tercapai.

Kuat digunakan untuk **tuning parameter** pada pencarian parameter **nonlinier/berisik** dan penuh **local minima**, karena bersifat pencarian global dan tidak memerlukan turunan.

---

### Tuning PID dengan GA
GA men-*tuning* \((K_p, K_i, K_d)\) dengan **fitness** yang mencerminkan tujuan kontrol, misalnya:

\[
\text{Fitness} \;=\; w_\text{IAE}\cdot \text{IAE}
\;+\; w_\text{OS}\cdot \text{Overshoot(\%)}
\;+\; w_\text{SET}\cdot \text{Settling Time}
\;+\; w_\text{ENERGY}\cdot \text{Energy}
\]

- **Bobot \(w\)** disetel sesuai prioritas (minim overshoot vs. cepat stabil vs. hemat energi).  
- **Kendala**: batas \(K_p, K_i, K_d\), anti-windup, penalti jika respons tak stabil.  
- **Stop**: **Fixed Generations** atau **Auto/Early-Stopping** (patience, ambang perbaikan \(\Delta\) fitness, atau target performa).

Hasilnya, GA mengeksplorasi kombinasi PID yang menyeimbangkan **kecepatan**, **stabilitas**, dan **efisiensi energi**, tanpa perlu model analitik yang sangat presisi—cocok untuk sistem riil yang kompleks.


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
