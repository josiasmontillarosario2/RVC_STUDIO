import os
import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None

def run_ffmpeg(input_path: str, out_dir: str, segment_time: int, sr: int, mono: bool, log_fn):
    base = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    out_pattern = os.path.join(out_dir, f"{base}_%03d.wav")

    cmd = [
        "ffmpeg",
        "-y",  # overwrite output files if they exist
        "-i", input_path,
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-ar", str(sr),
        "-ac", "1" if mono else "2",
        "-c:a", "pcm_s16le",
        out_pattern
    ]

    log_fn("Comando:\n" + " ".join(cmd) + "\n\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in proc.stdout:
        log_fn(line)

    code = proc.wait()
    return code, out_dir

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Segmenter (FFmpeg)")
        self.geometry("780x520")

        self.input_path = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.segment_time = tk.StringVar(value="45")
        self.sr = tk.StringVar(value="40000")
        self.mono = tk.BooleanVar(value=True)

        # Row 1: input
        frm1 = tk.Frame(self)
        frm1.pack(fill="x", padx=10, pady=8)

        tk.Label(frm1, text="Audio de entrada:").pack(anchor="w")
        row = tk.Frame(frm1)
        row.pack(fill="x")
        tk.Entry(row, textvariable=self.input_path).pack(side="left", fill="x", expand=True)
        tk.Button(row, text="Seleccionar...", command=self.pick_input).pack(side="left", padx=6)

        # Row 2: output folder
        frm2 = tk.Frame(self)
        frm2.pack(fill="x", padx=10, pady=8)

        tk.Label(frm2, text="Carpeta de salida:").pack(anchor="w")
        row2 = tk.Frame(frm2)
        row2.pack(fill="x")
        tk.Entry(row2, textvariable=self.out_dir).pack(side="left", fill="x", expand=True)
        tk.Button(row2, text="Elegir...", command=self.pick_outdir).pack(side="left", padx=6)

        # Row 3: options
        frm3 = tk.Frame(self)
        frm3.pack(fill="x", padx=10, pady=8)

        tk.Label(frm3, text="Opciones:").pack(anchor="w")
        opt = tk.Frame(frm3)
        opt.pack(fill="x")

        tk.Label(opt, text="Segmento (seg):").pack(side="left")
        tk.Entry(opt, width=6, textvariable=self.segment_time).pack(side="left", padx=6)

        tk.Label(opt, text="Sample rate:").pack(side="left")
        tk.Entry(opt, width=8, textvariable=self.sr).pack(side="left", padx=6)

        tk.Checkbutton(opt, text="Mono (ac=1)", variable=self.mono).pack(side="left", padx=10)

        self.btn_run = tk.Button(self, text="Segmentar", command=self.on_run)
        self.btn_run.pack(pady=6)

        # Log box
        tk.Label(self, text="Log:").pack(anchor="w", padx=10)
        self.log = tk.Text(self, height=18)
        self.log.pack(fill="both", expand=True, padx=10, pady=8)

        if not ffmpeg_exists():
            messagebox.showerror("FFmpeg no encontrado",
                                 "No se encontró ffmpeg en PATH.\nInstala FFmpeg o añade ffmpeg.exe al PATH.")
            self.btn_run.config(state="disabled")

    def pick_input(self):
        path = filedialog.askopenfilename(
            title="Selecciona un audio",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.wma"),
                ("Todos", "*.*")
            ]
        )
        if path:
            self.input_path.set(path)
            # default output: same folder as input + /segments/<base>
            base = os.path.splitext(os.path.basename(path))[0]
            default_out = os.path.join(os.path.dirname(path), "segments", base)
            self.out_dir.set(default_out)

    def pick_outdir(self):
        path = filedialog.askdirectory(title="Selecciona carpeta de salida")
        if path:
            self.out_dir.set(path)

    def log_write(self, text):
        self.log.insert("end", text)
        self.log.see("end")
        self.update_idletasks()

    def on_run(self):
        inp = self.input_path.get().strip()
        out = self.out_dir.get().strip()

        if not inp or not os.path.exists(inp):
            messagebox.showerror("Error", "Selecciona un archivo de audio válido.")
            return
        if not out:
            messagebox.showerror("Error", "Selecciona una carpeta de salida.")
            return

        try:
            seg = int(self.segment_time.get().strip())
            sr = int(self.sr.get().strip())
            if seg <= 0 or sr <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Segmento y sample rate deben ser números positivos.")
            return

        self.btn_run.config(state="disabled")
        self.log.delete("1.0", "end")
        self.log_write("Iniciando...\n\n")

        def worker():
            code, out_dir = run_ffmpeg(inp, out, seg, sr, self.mono.get(), self.log_write)
            if code == 0:
                messagebox.showinfo("Listo ✅", f"Segmentación completada.\nSalida:\n{out_dir}")
            else:
                messagebox.showerror("Error", f"FFmpeg falló (exit code {code}).\nRevisa el log.")
            self.btn_run.config(state="normal")

        threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    app = App()
    app.mainloop()
