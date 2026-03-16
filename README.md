# GPTinker2

A modern and simple desktop GUI for generating text with GPT-2 models locally.

## Features

- **5 model sizes**: DistilGPT-2 (82M) through GPT-2 XL (1.5B)
- **Real-time streaming**: watch tokens appear as they're generated
- **Adjustable parameters**: max tokens, temperature
- **Performance metrics**: tokens/sec, elapsed time, device info
- **Stop button**: interrupt generation at any time
- **GPU/CPU auto-detection**: uses CUDA if available

## Requirements

```bash
pip install transformers torch
```

## Usage

```bash
python gpt2_gui.py
```

Or for a minimal raw output demo:

```bash
python gpt2.py
```

Dark-themed UI built with Tkinter. Select a model, enter a prompt, and click Generate.
