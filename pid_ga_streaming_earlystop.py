# pid_ga_motor_speed_streaming_formula.py
import os, sys, uuid, csv, math, json, glob, traceback
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.integrate import odeint

# Matplotlib (plot)
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

# Gradio UI
import gradio as gr

# ------- Setup cache & log di folder lokal -------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["GRADIO_TEMP_DIR"] = os.path.join(SCRIPT_DIR, ".gradio_tmp")
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)

RUN_ROOT = os.path.join(SCRIPT_DIR, "motor_speed_runs")
os.makedirs(RUN_ROOT, exist_ok=True)

print("[INFO] Python:", sys.version)
print("[INFO] Gradio:", getattr(gr, "__version__", "unknown"))
print("[INFO] Script dir:", SCRIPT_DIR)
print("[INFO] GRADIO_TEMP_DIR:", os.environ["GRADIO_TEMP_DIR"])
print("[INFO] Run root:", RUN_ROOT)

# ===================== Model Motor & PID =====================
@dataclass
class MotorParams:
    J: float = 0.01  # moment of inertia
    b: float = 0.1   # damping ratio
    K: float = 0.01  # motor constant

def motor_speed(x, t, u, J, b, K):
    """
    x = [theta, omega]
    d(theta)/dt = omega
    d(omega)/dt = (u - b*omega - K*theta)/J
    """
    theta, omega = x
    dxdt = [omega, (u - b*omega - K*theta)/J]
    return dxdt

def pid_controller(y, setpoint, integral, last_error, dt, Kp, Ki, Kd):
    error = setpoint - y
    integral = integral + error * dt
    derivative = (error - last_error) / dt if dt > 0 else 0.0
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral, error

def metrics_from_trace(y, t, setpoint):
    y = np.asarray(y); t = np.asarray(t)
    rmse = float(np.sqrt(np.mean((setpoint - y) ** 2)))
    overshoot = float(max(0.0, (y.max() - setpoint) / max(1e-9, setpoint) * 100.0))
    idx90 = np.where(y >= 0.9 * setpoint)[0]
    rise_time = float(t[idx90[0]]) if idx90.size > 0 else 1e6
    band = 0.02 * setpoint
    outside = np.where(np.abs(y - setpoint) > band)[0]
    settling_time = float(t[outside[-1]]) if outside.size > 0 else 0.0
    return rmse, overshoot, rise_time, settling_time

def evaluate_one(indiv, t, setpoint, params: MotorParams):
    Kp, Ki, Kd = float(indiv[0]), float(indiv[1]), float(indiv[2])
    y0 = [0.0, 0.0]   # theta, omega
    y = [0.0]         # simpan omega sebagai output
    integral_err = 0.0
    last_err = 0.0
    dt = float(t[1] - t[0])
    for i in range(1, len(t)):
        u, integral_err, err = pid_controller(y[-1], setpoint, integral_err, last_err, dt, Kp, Ki, Kd)
        last_err = err
        next_state = odeint(motor_speed, y0, [t[i-1], t[i]], args=(u, params.J, params.b, params.K))[-1]
        y0 = next_state
        y.append(next_state[1])  # omega
    rmse, overshoot, rise_time, settling_time = metrics_from_trace(y, t, setpoint)
    return np.array(y, dtype=float), rmse, overshoot, rise_time, settling_time

# ===================== Fitness: formula dapat diubah =====================
def fitness_from_formula(rmse, overshoot, rise, settling, formula_text, fallback_weights=(0.7,0.1,0.1,0.1)):
    RMSE = float(rmse); OS = float(overshoot); Rise = float(rise); Settling = float(settling)
    safe_env = {
        "__builtins__": None,
        "RMSE": RMSE, "OS": OS, "Rise": Rise, "Settling": Settling,
        "abs": abs, "max": max, "min": min, "pow": pow,
        "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
    }
    try:
        val = eval(formula_text, {"__builtins__": None}, safe_env)
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            raise ValueError("invalid fitness")
        return float(val)
    except Exception:
        w_rmse, w_os, w_rt, w_ts = fallback_weights
        return float(w_rmse*RMSE + w_os*OS + w_rt*Rise + w_ts*Settling)

# ===================== GA Helper =====================
def clip_bounds(ind, bounds):
    (kpl, kpu), (kil, kiu), (kdl, kdu) = bounds
    ind[0] = float(np.clip(ind[0], kpl, kpu))
    ind[1] = float(np.clip(ind[1], kil, kiu))
    ind[2] = float(np.clip(ind[2], kdl, kdu))
    return ind

def selection(population, fitnesses, t_size=3, rng=np.random):
    N = len(population); k = min(t_size, N)
    idxs = rng.choice(np.arange(N), size=k, replace=False)
    best_idx = min(idxs, key=lambda i: fitnesses[i]["fitness"])  # fitness kecil = lebih baik
    return population[best_idx].copy()

def crossover(ind1, ind2, crossover_rate=0.25, rng=np.random):
    if rng.rand() < crossover_rate:
        point = rng.randint(1, len(ind1))
        c1 = np.concatenate([ind1[:point], ind2[point:]])
        c2 = np.concatenate([ind2[:point], ind1[point:]])
        return c1, c2
    return ind1.copy(), ind2.copy()

def mutate(ind, mutation_rate=0.75, rng=np.random):
    out = ind.copy()
    if rng.rand() < mutation_rate:
        mp = rng.randint(0, len(out))
        out[mp] += rng.normal(loc=-1.0, scale=1.0)
        out[mp] = max(0.0, out[mp])  # non-negatif (gaya Oskar)
    return out

# ===================== Util: auto-load last run =====================
def list_run_dirs():
    if not os.path.isdir(RUN_ROOT): return []
    return [os.path.join(RUN_ROOT, d) for d in os.listdir(RUN_ROOT)
            if os.path.isdir(os.path.join(RUN_ROOT, d)) and d.startswith("run_")]

def load_last_run(gallery_keep=20):
    dirs = list_run_dirs()
    if not dirs: return [], [], "_(Belum ada run tersimpan)_", ""
    last = max(dirs, key=os.path.getmtime)
    rows_path = os.path.join(last, "rows.json")
    csv_path  = os.path.join(last, "log_min_overshoot.csv")
    rows = []
    if os.path.exists(rows_path):
        try:
            with open(rows_path, "r") as f: rows = json.load(f)
        except Exception: rows = []
    elif os.path.exists(csv_path):
        try:
            import csv as _csv
            with open(csv_path, "r") as f:
                reader = _csv.reader(f); _ = next(reader, None)
                for r in reader:
                    try: gen = int(r[0])
                    except: gen = r[0]
                    rows.append([gen] + r[1:])
        except Exception: rows = []
    else:
        for p in sorted(glob.glob(os.path.join(last, "*_minOS.png"))):
            gen = os.path.basename(p).split("_")[0]
            try: gen = int(gen)
            except: pass
            rows.append([gen, "", "", "", os.path.relpath(p, start=SCRIPT_DIR), "", "", "", "", ""])
    imgs = sorted(glob.glob(os.path.join(last, "*_minOS.png")))[-int(gallery_keep):]
    imgs_rel = [os.path.relpath(p, start=SCRIPT_DIR) for p in imgs]
    msg = f"_Loaded last run:_ **{os.path.basename(last)}** — {len(imgs_rel)} image(s)"
    return rows, imgs_rel, msg, last

# ===================== Streaming GA =====================
def run_ga_streaming(
    # System
    J, b, K, setpoint, t_end, n_points,
    # GA (slider khusus)
    population_size, mode_stop, num_generations, mutation_rate, crossover_rate,
    # Early Stop
    patience, min_delta, min_generations, target_fitness, target_os,
    # Bounds
    kp_min, kp_max, ki_min, ki_max, kd_min, kd_max,
    # Fitness formula
    formula_text,
    # Logging & seed (state)
    gallery_keep, seed
):
    try:
        rng = np.random.default_rng(int(seed))
        params = MotorParams(J=float(J), b=float(b), K=float(K))
        setpoint = float(setpoint)
        t = np.linspace(0.0, float(t_end), int(n_points))
        bounds = ((float(kp_min), float(kp_max)),
                  (float(ki_min), float(ki_max)),
                  (float(kd_min), float(kd_max)))

        def rand_ind():
            return np.array([
                rng.uniform(bounds[0][0], bounds[0][1]),
                rng.uniform(bounds[1][0], bounds[1][1]),
                rng.uniform(bounds[2][0], bounds[2][1]),
            ], dtype=float)

        population = [rand_ind() for _ in range(int(population_size))]

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:6]
        run_dir = os.path.join(RUN_ROOT, "run_" + run_id)
        os.makedirs(run_dir, exist_ok=True)

        rows, gallery_items = [], []
        best_fit_so_far, no_improve = float("inf"), 0
        fixed_max_gen = int(num_generations) if mode_stop == "Fixed" else 10_000_000

        for gen in range(fixed_max_gen+1):
            fitnesses = []
            for idx, indiv in enumerate(population):
                indiv = clip_bounds(indiv, bounds)
                y, rmse, overshoot, rise_time, settling_time = evaluate_one(indiv, t, setpoint, params)
                fit = fitness_from_formula(rmse, overshoot, rise_time, settling_time, formula_text)
                fitnesses.append({
                    "idx": idx, "indiv": indiv.copy(), "y": y,
                    "rmse": rmse, "overshoot": overshoot,
                    "rise": rise_time, "settle": settling_time,
                    "fitness": fit,
                })

            best_fit_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i]["fitness"])
            bf = fitnesses[best_fit_idx]
            mo = min(fitnesses, key=lambda d: (d["overshoot"], d["fitness"]))  # min overshoot (tie break fitness)

            # ---------- Simpan gambar min-overshoot per generasi ----------
            fig_log = plt.figure(figsize=(4.8, 3.4))
            plt.plot(t, mo["y"], label="Output (omega)")
            plt.plot(t, np.ones_like(t)*setpoint, "--", linewidth=1.0, label="Setpoint")
            kp_str = f"Kp={mo['indiv'][0]:.3f}"; ki_str = f"Ki={mo['indiv'][1]:.3f}"; kd_str = f"Kd={mo['indiv'][2]:.3f}"
            os_str = f"Overshoot={mo['overshoot']:.3f}%"
            rt_str = f"Rise time={mo['rise']:.3f}s" if math.isfinite(mo["rise"]) else "Rise time=—"
            ts_str = f"Settling time={mo['settle']:.3f}s" if math.isfinite(mo["settle"]) else "Settling time=—"
            plt.plot([], [], " ", label=kp_str); plt.plot([], [], " ", label=ki_str); plt.plot([], [], " ", label=kd_str)
            plt.plot([], [], " ", label=os_str);  plt.plot([], [], " ", label=rt_str); plt.plot([], [], " ", label=ts_str)
            plt.legend(loc="lower right", fontsize=8, framealpha=0.9)
            plt.xlabel("Time (s)"); plt.ylabel("Speed (RPM)")
            plt.title(f"gen {gen} | minOS={mo['overshoot']:.3f}%")
            plt.tight_layout()
            fname = f"{gen:03d}_minOS.png"; fpath = os.path.join(run_dir, fname)
            plt.savefig(fpath, bbox_inches="tight"); plt.close(fig_log)

            row = [gen, float(mo["indiv"][0]), float(mo["indiv"][1]), float(mo["indiv"][2]),
                   os.path.relpath(fpath, start=SCRIPT_DIR),
                   float(mo["overshoot"]), float(mo["fitness"]), float(mo["rmse"]),
                   float(mo["rise"]) if math.isfinite(mo["rise"]) else "", float(mo["settle"])]
            rows.append(row)
            try:
                with open(os.path.join(run_dir, "rows.json"), "w") as jf:
                    json.dump(rows, jf)
            except Exception:
                pass

            gallery_items.append(fpath)
            if len(gallery_items) > int(gallery_keep):
                gallery_items = gallery_items[-int(gallery_keep):]

            # ---------- Plot live (best fitness) ----------
            fig_live = plt.figure(figsize=(8.2, 5.2))
            plt.plot(t, bf["y"], label="Output (omega)")
            plt.plot(t, np.ones_like(t)*setpoint, "--", label="Setpoint", linewidth=1.0)
            kp_b = f"Kp={bf['indiv'][0]:.3f}"; ki_b = f"Ki={bf['indiv'][1]:.3f}"; kd_b = f"Kd={bf['indiv'][2]:.3f}"
            os_b = f"Overshoot={bf['overshoot']:.3f}%"
            rt_b = f"Rise time={bf['rise']:.3f}s" if math.isfinite(bf["rise"]) else "Rise time=—"
            ts_b = f"Settling time={bf['settle']:.3f}s"
            plt.plot([], [], " ", label=kp_b); plt.plot([], [], " ", label=ki_b); plt.plot([], [], " ", label=kd_b)
            plt.plot([], [], " ", label=os_b);  plt.plot([], [], " ", label=rt_b); plt.plot([], [], " ", label=ts_b)
            plt.xlabel("Time (s)"); plt.ylabel("Speed (RPM)")
            plt.title(f"Best Fitness — Gen {gen} | Fit={bf['fitness']:.4f}  minOS(gen)={mo['overshoot']:.3f}%")
            plt.legend(loc="lower right", fontsize=9, framealpha=0.9); plt.grid(True, alpha=0.25); plt.tight_layout()

            status = (
                f"**Mode:** {mode_stop}  |  **Gen:** {gen}  |  **Best Fit:** {bf['fitness']:.4f}  |  "
                f"**minOS(gen):** {mo['overshoot']:.3f}%  |  "
                f"Best(Kp,Ki,Kd)=({bf['indiv'][0]:.3f}, {bf['indiv'][1]:.3f}, {bf['indiv'][2]:.3f})  \n"
            )

            yield fig_live, rows, gallery_items, status, run_dir

            # ---------- Kriteria berhenti ----------
            stop_reason = ""
            if mode_stop == "Fixed":
                if gen == int(num_generations):
                    stop_reason = "Stop: fixed generations tercapai."
            else:
                if bf["fitness"] < (best_fit_so_far - float(min_delta)):
                    best_fit_so_far = bf["fitness"]; no_improve = 0
                else:
                    no_improve += 1
                if (target_fitness and target_fitness > 0) and (bf["fitness"] <= target_fitness):
                    stop_reason = f"Stop: fitness <= target ({bf['fitness']:.4f} <= {target_fitness:.4f})"
                if (target_os and target_os > 0) and (mo["overshoot"] <= target_os):
                    stop_reason = (f"{stop_reason} & " if stop_reason else "") + \
                                  f"Stop: min overshoot <= target ({mo['overshoot']:.3f}% <= {target_os:.3f}%)"
                if (not stop_reason) and (gen >= int(min_generations)) and (no_improve >= int(patience)):
                    stop_reason = f"Stop: patience tercapai ({no_improve} generasi tanpa perbaikan)"

            if stop_reason:
                print("[INFO]", stop_reason)
                try:
                    path_csv = export_csv(rows, run_dir)
                    print("[INFO] Auto-saved CSV:", path_csv)
                except Exception:
                    pass
                break

            # ---------- GA step ----------
            new_pop = []
            while len(new_pop) < int(population_size):
                p1 = selection(population, fitnesses, t_size=min(3, len(population)), rng=np.random)
                p2 = selection(population, fitnesses, t_size=min(3, len(population)), rng=np.random)
                c1, c2 = crossover(p1, p2, crossover_rate=float(crossover_rate), rng=np.random)
                c1 = mutate(c1, mutation_rate=float(mutation_rate), rng=np.random)
                c2 = mutate(c2, mutation_rate=float(mutation_rate), rng=np.random)
                new_pop.extend([clip_bounds(c1, bounds), clip_bounds(c2, bounds)])
            population = new_pop[:int(population_size)]

    except Exception as e:
        traceback.print_exc()
        raise

# ===================== Export CSV =====================
def export_csv(rows, run_dir):
    try:
        if not rows: return None
        if not run_dir:
            run_dir = os.path.join(RUN_ROOT, "export_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "log_min_overshoot.csv")
        headers = ["Generation","Kp","Ki","Kd","filename","Overshoot(%)","Fitness","RMSE","RiseTime(s)","SettlingTime(s)"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f); writer.writerow(headers); writer.writerows(rows)
        return path
    except Exception:
        return None

# ===================== UI =====================
with gr.Blocks() as demo:
    gr.Markdown("# PID-GA — Motor Speed (Streaming) • Log Min Overshoot per Generasi • Legend • Formula Fitness Dapat Diubah • Auto-Load Run Terakhir")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### System Parameters")
            setpoint = gr.Number(value=1000.0, label="Setpoint (RPM)")
            J = gr.Number(value=0.01, label="J (moment of inertia)")
            b = gr.Number(value=0.1, label="b (damping ratio)")
            K = gr.Number(value=0.01, label="K (motor constant)")
            t_end = gr.Slider(1.0, 60.0, 10.0, step=0.5, label="t_end (s)")
            n_points = gr.Slider(200, 5000, 1000, step=50, label="Jumlah titik waktu")

        with gr.Column():
            gr.Markdown("### GA (slider khusus)")
            population_size = gr.Slider(4, 300, 20, step=1, label="Population size")
            mode_stop = gr.Dropdown(choices=["Fixed","Early Stopping"], value="Fixed", label="Mode berakhir")
            num_generations = gr.Slider(5, 1000, 50, step=1, label="Jumlah iterasi (Fixed mode)")
            crossover_rate = gr.Slider(0.0, 1.0, 0.25, step=0.01, label="Crossover rate")
            mutation_rate = gr.Slider(0.0, 1.0, 0.75, step=0.01, label="Mutation rate")

        with gr.Column():
            gr.Markdown("### Early Stopping (aktif bila Mode = Early Stopping)")
            patience = gr.Slider(1, 300, 30, step=1, label="Patience (generasi tanpa perbaikan)")
            min_delta = gr.Number(value=1e-3, label="Min Δ perbaikan fitness")
            min_generations = gr.Slider(0, 500, 30, step=1, label="Min generasi sebelum cek patience")
            target_fitness = gr.Number(value=0.0, label="Stop jika fitness ≤ (0=nonaktif)")
            target_os = gr.Number(value=0.0, label="Stop jika min overshoot ≤ (%) (0=nonaktif)")

        with gr.Column():
            gr.Markdown("### Bounds (Kp Ki Kd)")
            kp_min = gr.Number(value=0.0, label="Kp min"); kp_max = gr.Number(value=5.0, label="Kp max")
            ki_min = gr.Number(value=0.0, label="Ki min"); ki_max = gr.Number(value=1.0, label="Ki max")
            kd_min = gr.Number(value=0.0, label="Kd min"); kd_max = gr.Number(value=0.5, label="Kd max")

    gr.Markdown("### Fitness Function (ubah sesuka Anda)")
    formula_text = gr.Textbox(
        value="0.7*RMSE + 0.1*OS + 0.1*Rise + 0.1*Settling",
        lines=2,
        label="Rumus fitness (variabel: RMSE, OS, Rise, Settling; fungsi: abs, max, min, pow, sqrt, log, exp)"
    )

    # Tombol & komponen output
    run_btn = gr.Button("Run (streaming)")
    live_plot = gr.Plot()
    table_headers = ["Generation","Kp","Ki","Kd","filename","Overshoot(%)","Fitness","RMSE","RiseTime(s)","SettlingTime(s)"]
    table = gr.Dataframe(headers=table_headers, interactive=False)
    gallery = gr.Gallery(label="Image Output Log — min overshoot per generation", columns=6)
    status = gr.Markdown()

    # State untuk ekspor & nilai konstan (pengganti slider tersembunyi)
    state_rows = gr.State([])
    state_run_dir = gr.State("")
    state_gallery_keep = gr.State(20)  # simpan N gambar terakhir di galeri
    state_seed = gr.State(42)          # seed RNG

    # Auto-load run terakhir saat app dibuka
    def _on_load(gkeep):
        rows, imgs, msg, last_dir = load_last_run(gkeep)
        return rows, imgs, msg, rows, last_dir

    demo.load(_on_load, inputs=[state_gallery_keep], outputs=[table, gallery, status, state_rows, state_run_dir])

    # Streaming callback (generator)
    def _wrap_stream(*args):
        for fig, rows, gal, stat, run_dir in run_ga_streaming(*args):
            yield fig, rows, gal, stat, rows, run_dir

    run_btn.click(
        _wrap_stream,
        inputs=[
            J, b, K, setpoint, t_end, n_points,
            population_size, mode_stop, num_generations, mutation_rate, crossover_rate,
            patience, min_delta, min_generations, target_fitness, target_os,
            kp_min, kp_max, ki_min, ki_max, kd_min, kd_max,
            formula_text,
            state_gallery_keep, state_seed
        ],
        outputs=[live_plot, table, gallery, status, state_rows, state_run_dir]
    )

    gr.Markdown("### Export")
    export_btn = gr.Button("Export CSV (table)")
    export_file = gr.File(label="Download CSV")

    def _export(rows, run_dir):
        path = export_csv(rows, run_dir)
        return path

    export_btn.click(_export, inputs=[state_rows, state_run_dir], outputs=[export_file])

if __name__ == "__main__":
    demo.launch(
        allowed_paths=[RUN_ROOT, os.environ["GRADIO_TEMP_DIR"]],
        server_name="127.0.0.1",
        server_port=7860,
        debug=True,
        show_error=True
    )
