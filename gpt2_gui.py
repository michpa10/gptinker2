import threading
import time
import tkinter as tk
from tkinter import ttk

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
import torch

# ── colours ───────────────────────────────────────────────────────────────────
BG        = "#1e1e2e"
BG_WIDGET = "#2a2a3d"
BG_OUTPUT = "#13131f"
FG        = "#cdd6f4"
FG_DIM    = "#6c7086"
FG_PROMPT = "#cdd6f4"
FG_MODEL  = "#a6e3a1"
ACCENT    = "#89b4fa"
CURSOR    = "#f5c2e7"

# ── Model registry ───────────────────────────────────────────────────────────
MODELS = {
    "DistilGPT-2 (82M)":   "distilbert/distilgpt2",
    "GPT-2 Small (117M)":  "openai-community/gpt2",
    "GPT-2 Medium (345M)": "openai-community/gpt2-medium",
    "GPT-2 Large (774M)":  "openai-community/gpt2-large",
    "GPT-2 XL (1.5B)":     "openai-community/gpt2-xl",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_current_model_id = None
tokenizer = None
model = None
_stop_event = threading.Event()

# Token batch buffer — poll thread writes, main thread flushes on a timer
_token_buffer = []
_token_buffer_lock = threading.Lock()


class _StopOnEvent(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return _stop_event.is_set()


def load_model(model_id):
    global tokenizer, model, _current_model_id
    if model_id == _current_model_id:
        return
    # Unload existing model to free VRAM before loading the new one
    if model is not None:
        del model
        model = None
    if tokenizer is not None:
        del tokenizer
        tokenizer = None
    _current_model_id = None
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=DEVICE,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    _current_model_id = model_id


def _set_loading(active):
    if active:
        progress_bar.grid()
        progress_bar.start(12)
    else:
        progress_bar.stop()
        progress_bar.grid_remove()


def _flush_token_buffer():
    """Called on main thread every 50 ms during generation to drain buffered tokens."""
    with _token_buffer_lock:
        if _token_buffer:
            text = "".join(_token_buffer)
            _token_buffer.clear()
        else:
            text = None

    if text:
        output_box.insert(tk.END, text, "model")
        # Only scroll if already near the bottom
        yview_end = output_box.yview()[1]
        if yview_end >= 0.95:
            output_box.see(tk.END)

    if not _stop_event.is_set() and generate_btn.cget("text") in ("Stop", "Stopping…"):
        root.after(50, _flush_token_buffer)


def stop_generation():
    _stop_event.set()
    generate_btn.config(state="disabled", text="Stopping…")


def generate():
    prompt = prompt_box.get("1.0", "end-1c").strip()
    if not prompt:
        return

    _stop_event.clear()
    with _token_buffer_lock:
        _token_buffer.clear()

    generate_btn.config(text="Stop", command=stop_generation)
    output_box.config(state="normal")
    output_box.delete("1.0", tk.END)

    # Show prompt immediately; tokenize once and reuse
    output_box.insert(tk.END, prompt, "prompt")

    info_var.set("")

    def _run():
        load_model(MODELS[model_var.get()])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        root.after(0, lambda: (_set_loading(False), _start_generation(inputs)))

    _set_loading(True)
    threading.Thread(target=_run, daemon=True).start()


def _start_generation(inputs):
    info_var.set("")
    _do_generate(inputs)


def _do_generate(inputs):
    prompt_tokens = inputs["input_ids"].shape[1]
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    temp = float(temp_var.get())
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=int(tokens_var.get()),
        do_sample=temp > 0.0,
        temperature=temp if temp > 0.0 else 1.0,
        top_k=50,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([_StopOnEvent()]),
    )

    t0 = time.perf_counter()
    new_tokens_count = [0]

    def run():
        with torch.no_grad():
            model.generate(**gen_kwargs)

    threading.Thread(target=run, daemon=True).start()

    # Start the UI flush timer
    root.after(50, _flush_token_buffer)

    def poll():
        try:
            for token_text in streamer:
                if _stop_event.is_set():
                    break
                with _token_buffer_lock:
                    _token_buffer.append(token_text)
                new_tokens_count[0] += 1
        finally:
            # Flush any remaining tokens then finish
            elapsed = time.perf_counter() - t0
            n = new_tokens_count[0]
            tok_per_sec = n / elapsed if elapsed > 0 else 0
            info = (
                f"Prompt tokens: {prompt_tokens}  |  "
                f"New tokens: {n}  |  "
                f"Total: {prompt_tokens + n}  |  "
                f"{tok_per_sec:.1f} tok/s  |  "
                f"{elapsed:.2f}s  |  "
                f"{DEVICE.upper()} {str(model.dtype).replace('torch.', '')}"
            )
            root.after(0, lambda: finish(info))

    threading.Thread(target=poll, daemon=True).start()


def finish(info):
    # Final flush of any remaining buffered tokens
    with _token_buffer_lock:
        if _token_buffer:
            text = "".join(_token_buffer)
            _token_buffer.clear()
            output_box.insert(tk.END, text, "model")

    output_box.config(state="disabled")
    generate_btn.config(state="normal", text="Generate", command=generate)
    info_var.set(("(stopped)  " if _stop_event.is_set() else "") + info)


def on_key(event):
    if event.keysym == "Return" and (event.state & 0x4 or event.state & 0x1):
        generate()
        return "break"


# ── Window ────────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("GPT-2 Text Generator")
root.resizable(True, True)
root.geometry("1050x780")
root.configure(bg=BG)

style = ttk.Style(root)
style.theme_use("clam")
style.configure(".",
    background=BG, foreground=FG,
    fieldbackground=BG_WIDGET, bordercolor=BG_WIDGET,
    troughcolor=BG_WIDGET, selectbackground=ACCENT,
    selectforeground=BG,
)
style.configure("TFrame", background=BG)
style.configure("TLabel", background=BG, foreground=FG)
style.configure("TButton",
    background=ACCENT, foreground=BG,
    borderwidth=0, focusthickness=0,
)
style.map("TButton",
    background=[("disabled", BG_WIDGET), ("active", "#b4cef7")],
    foreground=[("disabled", FG_DIM)],
)
style.configure("TSpinbox", fieldbackground=BG_WIDGET, foreground=FG,
                arrowcolor=FG)
style.configure("TScale", troughcolor=BG_WIDGET, background=ACCENT)
style.configure("TScrollbar", background=BG_WIDGET, troughcolor=BG,
                arrowcolor=FG_DIM, bordercolor=BG)
style.configure("TCombobox", fieldbackground=BG_WIDGET, foreground=FG,
                background=BG_WIDGET, arrowcolor=FG, selectbackground=ACCENT)
style.map("TCombobox",
    fieldbackground=[("readonly", BG_WIDGET)],
    foreground=[("readonly", FG)],
)

PAD = {"padx": 12, "pady": 6}

frame = ttk.Frame(root, padding=16)
frame.grid(sticky="nsew")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)

# ── Prompt ────────────────────────────────────────────────────────────────────
ttk.Label(frame, text="Prompt").grid(row=0, column=0, sticky="nw", **PAD)
prompt_box = tk.Text(
    frame, width=60, height=5, wrap="word",
    font=("Monospace", 13), relief="flat",
    background=BG_WIDGET, foreground=FG_PROMPT,
    insertbackground=CURSOR, selectbackground=ACCENT,
    undo=True, borderwidth=0,
)
prompt_box.insert("1.0", "Once upon a time")
prompt_box.grid(row=0, column=1, columnspan=2, sticky="ew", **PAD)
prompt_box.bind("<Key>", on_key)

ttk.Label(frame, text="Ctrl+Enter or Shift+Enter to generate",
          foreground=FG_DIM, font=("Helvetica", 9)).grid(
    row=1, column=1, sticky="w", padx=12, pady=(0, 4))

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl_frame = ttk.Frame(frame)
ctrl_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=4)

ttk.Label(ctrl_frame, text="Model").pack(side="left")
model_var = tk.StringVar(value=next(iter(MODELS)))
model_dropdown = ttk.Combobox(ctrl_frame, textvariable=model_var,
                              values=list(MODELS.keys()),
                              state="readonly", width=22)
model_dropdown.pack(side="left", padx=(4, 16))
model_dropdown.bind("<<ComboboxSelected>>", lambda e: root.focus())
model_dropdown.bind("<FocusOut>", lambda e: model_dropdown.selection_clear())

ttk.Label(ctrl_frame, text="Max tokens").pack(side="left")
tokens_var = tk.IntVar(value=50)
ttk.Spinbox(ctrl_frame, from_=10, to=500, increment=10,
            textvariable=tokens_var, width=6).pack(side="left", padx=(4, 16))

ttk.Label(ctrl_frame, text="Temperature").pack(side="left")
temp_var = tk.DoubleVar(value=0.7)
temp_scale = ttk.Scale(ctrl_frame, from_=0.0, to=2.0, variable=temp_var,
                       orient="horizontal", length=160)
temp_scale.pack(side="left", padx=(4, 4))

temp_display = tk.StringVar(value="0.70")
def _update_temp_label(_=None):
    temp_display.set(f"{temp_var.get():.2f}")
temp_scale.bind("<Motion>", _update_temp_label)
temp_scale.bind("<ButtonRelease-1>", _update_temp_label)

ttk.Label(ctrl_frame, textvariable=temp_display, width=4,
          foreground=ACCENT).pack(side="left", padx=(0, 16))

generate_btn = ttk.Button(ctrl_frame, text="Generate", command=generate)
generate_btn.pack(side="left")

# ── Output ────────────────────────────────────────────────────────────────────
ttk.Label(frame, text="Output").grid(row=3, column=0, sticky="nw", **PAD)
output_box = tk.Text(
    frame, width=60, height=12, wrap="word",
    state="disabled", relief="flat",
    background=BG_OUTPUT, foreground=FG,
    font=("Monospace", 13),
    selectbackground=ACCENT, borderwidth=0,
)
output_box.tag_configure("prompt", foreground=FG_PROMPT)
output_box.tag_configure("model",  foreground=FG_MODEL)
output_box.grid(row=3, column=1, columnspan=2, sticky="nsew", **PAD)
frame.rowconfigure(3, weight=1)

scrollbar = ttk.Scrollbar(frame, orient="vertical", command=output_box.yview)
scrollbar.grid(row=3, column=3, sticky="ns", pady=6)
output_box.config(yscrollcommand=scrollbar.set)

# ── Progress bar (hidden until loading) ──────────────────────────────────────
progress_bar = ttk.Progressbar(frame, mode="indeterminate")
progress_bar.grid(row=4, column=0, columnspan=4, sticky="ew", padx=12, pady=(6, 0))
progress_bar.grid_remove()

# ── Info bar ──────────────────────────────────────────────────────────────────
info_var = tk.StringVar()
ttk.Label(frame, textvariable=info_var, foreground=FG_DIM,
          font=("Helvetica", 9), anchor="w").grid(
    row=5, column=0, columnspan=4, sticky="ew", padx=12, pady=(2, 0))

# ── Load default model in background so UI appears immediately ────────────────
def _initial_load():
    load_model(next(iter(MODELS.values())))
    root.after(0, lambda: _set_loading(False))
_set_loading(True)
threading.Thread(target=_initial_load, daemon=True).start()

root.mainloop()
