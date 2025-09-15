# pid_ga_streaming_earlystop.py
import os, sys, uuid, csv, math, traceback, json, glob
from dataclasses import dataclass
from datetime import datetime
import numpy as np

# ---- Paksa cache Gradio di sebelah script (hindari error /tmp permission) ----
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["GRADIO_TEMP_DIR"] = os.path.join(SCRIPT_DIR, ".gradio_tmp")
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

import gradio as gr

print("[INFO] Python:", sys.version)
print("[INFO] Gradio:", getattr(gr, "__version__", "unknown"))
print("[INFO] Script dir:", SCRIPT_DIR)
print("[INFO] GRADIO_TEMP_DIR:", os.environ["GRADIO_TEMP_DIR"])

# ---- Folder log di sebelah script ----
ROOT_DIR = os.path.join(SCRIPT_DIR, "pid_ga_runs")
os.makedirs(ROOT_DIR, exist_ok=True)
print("[INFO] Log dir:", ROOT_DIR)

# ---------------- Plant & Simulation ----------------
@dataclass
class PlantParams:
    wn: float = 1.6
    zeta: float = 0.35
    K: float = 1.0

def simulate_pid_response(Kp, Ki, Kd, T=8.0, dt=0.01, setpoint=1.0, sat=10.0, params=PlantParams()):
    N = int(T/dt)
    x1 = 0.0; x2 = 0.0
    I = 0.0
    prev_e = setpoint - x1
    y = np.zeros(N, dtype=float)
    u = np.zeros(N, dtype=float)
    t = np.linspace(0, T, N)
    for i in range(N):
        y[i] = x1
        e = setpoint - y[i]
        I += e*dt
        D = (e - prev_e)/dt if i>0 else 0.0
        ctrl = Kp*e + Ki*I + Kd*D
        ctrl = np.clip(ctrl, -sat, sat)  # saturasi
        prev_e = e
        # plant orde-2
        x1_dot = x2
        x2_dot = -2*params.zeta*params.wn*x2 - (params.wn**2)*x1 + params.K*ctrl
        x1 += x1_dot*dt
        x2 += x2_dot*dt
        u[i] = ctrl
    # metrik
    e = setpoint - y
    iae = float(np.sum(np.abs(e))*dt)
    rmse = float(np.sqrt(np.mean(e**2)))
    overshoot = float(max(0.0, (y.max()-setpoint)/max(1e-9,setpoint)*100.0))
    # settling time (2%)
    band = 0.02*setpoint
    settling_idx = len(t)-1
    for i in range(len(t)-1, -1, -1):
        if abs(y[i]-setpoint) > band:
            settling_idx = i
            break
    settling = float(t[settling_idx])
    # rise time 10–90%
    t10 = next((t[i] for i in range(len(t)) if y[i] >= 0.1*setpoint), None)
    t90 = next((t[i] for i in range(len(t)) if y[i] >= 0.9*setpoint), None)
    risetime = float(t90 - t10) if t10 is not None and t90 is not None and t90>=t10 else float("nan")
    energy = float(np.sum(u**2)*dt)
    return dict(t=t, y=y, u=u, iae=iae, rmse=rmse, overshoot=overshoot, settling=settling, risetime=risetime, energy=energy)

# ---------------- GA Helpers ----------------
def fitness(m, w):
    # minimisasi
    val = w[0]*m["iae"] + w[1]*m["overshoot"] + w[2]*m["settling"] + w[3]*m["energy"]
    if np.isnan(val) or np.isinf(val):
        return 1e9
    # penalti lembut jika overshoot sangat besar
    if m["overshoot"] > 50:
        val *= (1 + 0.01*(m["overshoot"]-50))
    return float(val)

def rand_pid(bounds):
    (kpl, kpu), (kil, kiu), (kdl, kdu) = bounds
    return np.array([
        np.random.uniform(kpl, kpu),
        np.random.uniform(kil, kiu),
        np.random.uniform(kdl, kdu),
    ], dtype=float)

def clip_pid(pid, bounds):
    (kpl, kpu), (kil, kiu), (kdl, kdu) = bounds
    pid[0] = np.clip(pid[0], kpl, kpu)
    pid[1] = np.clip(pid[1], kil, kiu)
    pid[2] = np.clip(pid[2], kdl, kdu)
    return pid

def tournament_select(pop, fits, k=3):
    idxs = np.random.choice(len(pop), size=k, replace=False)
    best = int(idxs[0]); bestf = float(fits[best])
    for i in idxs[1:]:
        if fits[i] < bestf:
            best = int(i); bestf = float(fits[i])
    return pop[best].copy()

def crossover(p1, p2):
    a = np.random.uniform(0.25, 0.75, size=3)
    return a*p1 + (1-a)*p2

def mutate(ind, bounds, scale=0.1):
    span = np.array([bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0], bounds[2][1]-bounds[2][0]], dtype=float)
    return clip_pid(ind + np.random.randn(3)*scale*span, bounds)

# ---------------- Util: Auto-load last run ----------------
def list_run_dirs():
    if not os.path.isdir(ROOT_DIR):
        return []
    dirs = []
    for d in os.listdir(ROOT_DIR):
        p = os.path.join(ROOT_DIR, d)
        if os.path.isdir(p) and d.startswith("run_"):
            dirs.append(p)
    return dirs

def load_last_run(gallery_keep=20):
    dirs = list_run_dirs()
    if not dirs:
        return [], [], "_(Belum ada run tersimpan)_", ""
    # pilih yang terbaru berdasarkan mtime folder
    last = max(dirs, key=os.path.getmtime)
    rows_path = os.path.join(last, "rows.json")
    csv_path  = os.path.join(last, "log_min_overshoot.csv")

    rows = []
    if os.path.exists(rows_path):
        try:
            with open(rows_path, "r") as f:
                rows = json.load(f)
        except Exception:
            rows = []
    elif os.path.exists(csv_path):
        try:
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                hdr = next(reader, None)  # skip header
                for r in reader:
                    # coerce types where possible, else keep string
                    try:
                        gen = int(r[0])
                    except: gen = r[0]
                    rows.append([gen] + r[1:])
        except Exception:
            rows = []
    else:
        # fallback: konstruksi minimal dari gambar
        imgs = sorted(glob.glob(os.path.join(last, "*_minOS.png")))
        for p in imgs:
            gen = os.path.basename(p).split("_")[0]
            try: gen = int(gen)
            except: pass
            rows.append([gen, "", "", "", os.path.relpath(p, start=SCRIPT_DIR), "", "", "", "", ""])

    # siapkan galeri (maks gallery_keep)
    imgs = sorted(glob.glob(os.path.join(last, "*_minOS.png")))
    imgs = imgs[-int(gallery_keep):]
    imgs_rel = [os.path.relpath(p, start=SCRIPT_DIR) for p in imgs]
    msg = f"_Loaded last run:_ **{os.path.basename(last)}** — {len(imgs_rel)} image(s)"
    return rows, imgs_rel, msg, last

# ---------------- Streaming GA + Early Stop ----------------
def run_ga_streaming(
    kp_min, kp_max, ki_min, ki_max, kd_min, kd_max,
    pop_size, generations, crossover_rate, mutation_rate, elite,
    w_iae, w_os, w_set, w_energy,
    mode_stop, patience, min_delta, min_generations, target_fitness, target_os,
    seed, gallery_keep
):
    try:
        np.random.seed(int(seed))
        bounds = ((kp_min, kp_max), (ki_min, ki_max), (kd_min, kd_max))
        weights = (w_iae, w_os, w_set, w_energy)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:6]
        run_dir = os.path.join(ROOT_DIR, "run_"+run_id)
        os.makedirs(run_dir, exist_ok=True)

        # init populasi
        pop = [rand_pid(bounds) for _ in range(pop_size)]

        # state
        rows = []          # tabel (1 baris per generasi utk min overshoot)
        gallery_items = [] # list path gambar untuk Gallery
        best_fit_so_far = float("inf")
        no_improve = 0

        def eval_one(ind):
            m = simulate_pid_response(*ind)
            return fitness(m, weights), m

        # evaluasi awal
        fits, mets = [], []
        for ind in pop:
            f, m = eval_one(ind); fits.append(f); mets.append(m)

        max_gens = int(generations) if mode_stop == "Fixed Generations" else 10_000_000
        for gen in range(max_gens+1):
            # --- log kandidat dengan overshoot terkecil (tie-break by fitness) ---
            overs = np.array([m["overshoot"] for m in mets], dtype=float)
            idx_os_min = int(np.argmin(overs))
            min_os = overs[idx_os_min]
            tie_idxs = np.where(overs == min_os)[0]
            if len(tie_idxs) > 1:
                best_fit = float("inf"); best_idx = idx_os_min
                for i in tie_idxs:
                    if fits[i] < best_fit:
                        best_fit = float(fits[i]); best_idx = int(i)
                idx_os_min = best_idx

            m = mets[idx_os_min]; indv = pop[idx_os_min]
            # simpan gambar min overshoot per generasi
            fig_log = plt.figure(figsize=(4.5,3))
            plt.plot(m["t"], m["y"], label="Output")
            plt.plot(m["t"], np.ones_like(m["t"]), "--", linewidth=1.0, label="Setpoint")
            plt.legend(loc="lower right", fontsize=7)
            plt.title("gen {} | minOS={:.3f}%".format(gen, m["overshoot"]))
            plt.tight_layout()
            fname = "{}_minOS.png".format(str(gen).zfill(3))
            fpath = os.path.join(run_dir, fname)
            plt.savefig(fpath, bbox_inches="tight"); plt.close(fig_log)

            # update table & gallery
            row = [
                gen, float(indv[0]), float(indv[1]), float(indv[2]),
                os.path.relpath(fpath, start=SCRIPT_DIR),
                float(m["overshoot"]), float(fits[idx_os_min]), float(m["rmse"]),
                float(m["risetime"]) if math.isfinite(m["risetime"]) else "",
                float(m["settling"]),
            ]
            rows.append(row)
            # persist rows setiap generasi → agar bisa auto-load saat restart
            try:
                with open(os.path.join(run_dir, "rows.json"), "w") as jf:
                    json.dump(rows, jf)
            except Exception:
                pass

            gallery_items.append(fpath)
            if len(gallery_items) > int(gallery_keep):
                gallery_items = gallery_items[-int(gallery_keep):]

            # best-of-generation untuk live plot & tracking early stop
            best_i = int(np.argmin(fits))
            best_m = mets[best_i]; best_ind = pop[best_i]
            best_fit = float(fits[best_i])

            # early stop (hanya untuk mode Auto)
            stop_reason = ""
            if mode_stop == "Auto (Best found)":
                if best_fit < best_fit_so_far - float(min_delta):
                    best_fit_so_far = best_fit
                    no_improve = 0
                else:
                    no_improve += 1

                # target opsional
                if (target_fitness is not None) and (target_fitness > 0) and (best_fit <= target_fitness):
                    stop_reason = "Stop: fitness <= target ({:.4f} <= {:.4f})".format(best_fit, target_fitness)
                if (target_os is not None) and (target_os > 0) and (min_os <= target_os):
                    if stop_reason:
                        stop_reason += " & "
                    stop_reason += "Stop: min overshoot <= target ({:.3f}% <= {:.3f}%)".format(min_os, target_os)

                # patience setelah min_generations
                if (not stop_reason) and (gen >= int(min_generations)) and (no_improve >= int(patience)):
                    stop_reason = "Stop: patience reached ({} gens no improvement)".format(no_improve)

            # live plot (best fitness)
            fig_live = plt.figure(figsize=(8,5))
            plt.plot(best_m["t"], best_m["y"], label="Output")
            plt.plot(best_m["t"], np.ones_like(best_m["t"]), "--", label="Setpoint", linewidth=1.0)
            plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
            plt.title("Best Fitness — Gen {} | Kp={:.3f}, Ki={:.3f}, Kd={:.3f}, Fit={:.4f}, minOS={:.3f}%".format(
                gen, best_ind[0], best_ind[1], best_ind[2], best_fit, min_os
            ))
            plt.legend(); plt.tight_layout()

            status = "**Mode:** {}  |  **Gen:** {}  |  **Best Fit:** {:.4f}  |  **minOS:** {:.3f}%  |  **No-improve:** {}\n{}".format(
                mode_stop, gen, best_fit, min_os, no_improve, stop_reason
            )

            # stream state (kembalikan juga run_dir agar Export tahu foldernya)
            yield fig_live, rows, gallery_items, status, run_dir

            # berhenti?
            if (mode_stop == "Fixed Generations" and gen == int(generations)) or \
               (mode_stop == "Auto (Best found)" and stop_reason):
                # simpan CSV otomatis juga saat berhenti
                try:
                    path_csv, _ = export_csv(rows, run_dir)
                    print("[INFO] Auto-saved CSV:", path_csv)
                except Exception:
                    pass
                break

            # GA step generasi berikutnya
            idx_sorted = np.argsort(fits)
            elite_idx = idx_sorted[:int(elite)]
            new_pop = [pop[i].copy() for i in elite_idx]
            while len(new_pop) < pop_size:
                p1 = tournament_select(pop, fits)
                p2 = tournament_select(pop, fits)
                child = crossover(p1, p2) if np.random.rand() < crossover_rate else p1.copy()
                if np.random.rand() < mutation_rate:
                    child = mutate(child, bounds)
                new_pop.append(clip_pid(child, bounds))
            pop = new_pop
            # re-evaluate
            fits, mets = [], []
            for ind in pop:
                f, m = eval_one(ind); fits.append(f); mets.append(m)

    except Exception as e:
        traceback.print_exc()
        raise

# ---------------- Export helper ----------------
def export_csv(rows, run_dir):
    try:
        if not rows:
            return None, "Tidak ada data untuk diekspor."
        if not run_dir:
            run_dir = os.path.join(ROOT_DIR, "export_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "log_min_overshoot.csv")
        headers = ["Generation","Kp","Ki","Kd","filename","Overshoot","Fitness","RMSE","RiseTime","SettlingTime"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        return path, "CSV tersimpan di: {}".format(path)
    except Exception as e:
        return None, "Gagal ekspor CSV: {}".format(e)

# ---------------- UI ----------------
with gr.Blocks() as demo:
    gr.Markdown("# PID-GA — Streaming + Early Stop (Min Overshoot Logger) — Auto-load last run")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Bounds PID")
            kp_min = gr.Slider(0, 20, 0, label="Kp min")
            kp_max = gr.Slider(0, 20, 10, label="Kp max")
            ki_min = gr.Slider(0, 20, 0, label="Ki min")
            ki_max = gr.Slider(0, 20, 6, label="Ki max")
            kd_min = gr.Slider(0, 10, 0, label="Kd min")
            kd_max = gr.Slider(0, 10, 3, label="Kd max")
        with gr.Column():
            gr.Markdown("### GA Config")
            pop_size = gr.Slider(10, 300, 60, step=1, label="Population")
            generations = gr.Slider(10, 500, 100, step=1, label="Generations (fixed mode)")
            crossover_rate = gr.Slider(0.1, 1.0, 0.85, step=0.01, label="Crossover rate")
            mutation_rate = gr.Slider(0.0, 0.5, 0.10, step=0.01, label="Mutation rate")
            elite = gr.Slider(0, 20, 3, step=1, label="Elites kept")
            seed = gr.Number(value=42, label="Random seed")
        with gr.Column():
            gr.Markdown("### Fitness Weights")
            w_iae = gr.Slider(0.0, 2.0, 1.0, step=0.01, label="w: IAE")
            w_os = gr.Slider(0.0, 1.0, 0.25, step=0.01, label="w: Overshoot")
            w_set = gr.Slider(0.0, 1.0, 0.25, step=0.01, label="w: Settling time")
            w_energy = gr.Slider(0.0, 0.5, 0.05, step=0.01, label="w: Energy")
        with gr.Column():
            gr.Markdown("### Stop Mode")
            mode_stop = gr.Dropdown(choices=["Fixed Generations", "Auto (Best found)"], value="Fixed Generations", label="Mode berakhir")
            patience = gr.Slider(1, 200, 30, step=1, label="Patience (auto mode)")
            min_delta = gr.Number(value=0.001, label="Min improvement Δfitness (auto)")
            min_generations = gr.Slider(0, 300, 30, step=1, label="Min generations before patience (auto)")
            target_fitness = gr.Number(value=0.0, label="Stop if fitness ≤ (0 = disable)")
            target_os = gr.Number(value=0.0, label="Stop if min overshoot ≤ (%) (0 = disable)")
        with gr.Column():
            gr.Markdown("### Logging")
            gallery_keep = gr.Slider(1, 60, 20, step=1, label="Keep last N images in gallery")

    run_btn = gr.Button("Run (streaming)")
    live_plot = gr.Plot()
    table_headers = ["Generation","Kp","Ki","Kd","filename","Overshoot","Fitness","RMSE","RiseTime","SettlingTime"]
    table = gr.Dataframe(headers=table_headers, interactive=False)
    gallery = gr.Gallery(label="Image Output Log — min overshoot per generation", columns=6)
    status = gr.Markdown()

    # state untuk Export
    state_rows = gr.State([])
    state_run_dir = gr.State("")

    # === Auto-load last run saat app dibuka ===
    def _on_load(gkeep):
        rows, imgs, msg, last_dir = load_last_run(gkeep)
        return rows, imgs, msg, rows, last_dir

    demo.load(_on_load, inputs=[gallery_keep], outputs=[table, gallery, status, state_rows, state_run_dir])

    # streaming callback: terus menerus yield update + simpan state
    def _wrap_stream(*args):
        for fig, rows, gal, stat, run_dir in run_ga_streaming(*args):
            yield fig, rows, gal, stat, rows, run_dir

    run_btn.click(
        _wrap_stream,
        inputs=[kp_min,kp_max,ki_min,ki_max,kd_min,kd_max,
                pop_size,generations,crossover_rate,mutation_rate,elite,
                w_iae,w_os,w_set,w_energy,
                mode_stop,patience,min_delta,min_generations,target_fitness,target_os,
                seed,gallery_keep],
        outputs=[live_plot, table, gallery, status, state_rows, state_run_dir]
    )

    gr.Markdown("### Export")
    export_btn = gr.Button("Export CSV (table)")
    export_file = gr.File(label="Download CSV")

    def _export(rows, run_dir):
        path, msg = export_csv(rows, run_dir)
        return path  # Gradio akan sediakan tombol unduh

    export_btn.click(_export, inputs=[state_rows, state_run_dir], outputs=[export_file])

if __name__ == "__main__":
    try:
        demo.launch(
            allowed_paths=[ROOT_DIR, os.environ["GRADIO_TEMP_DIR"]],
            server_name="127.0.0.1",
            server_port=7860,
            debug=True,
            show_error=True
        )
    except TypeError:
        demo.launch(server_name="127.0.0.1", server_port=7860, debug=True, show_error=True)
