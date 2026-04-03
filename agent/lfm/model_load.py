#!/usr/bin/env python3
import os
import sys
import socket
import platform
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "LiquidAI/LFM2.5-350M"
PROMPT = '''You are an assistant helping me communicate professionally.

Task:
Write 3 WhatsApp message variants to inform my manager that I will be 10 minutes late to today’s meeting.

Requirements:
1) Tone variants:
   - Variant A: Formal and concise
   - Variant B: Friendly and polite
   - Variant C: Very brief (one sentence)
2) Each variant must be between 18 and 35 words, except Variant C which must be <= 14 words.
3) Include a short apology in each variant.
4) Mention that I am already on the way.
5) Do not use emojis.
6) Do not use placeholders like [Name]; write as if sending directly.
7) After the 3 variants, add a section titled "Best Choice" and pick one variant with a one-line reason.

Output format (must follow exactly):
Variant A: ...
Variant B: ...
Variant C: ...
Best Choice: Variant X — <one line reason>'''

def has_internet(host="huggingface.co", port=443, timeout=3):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False

def tune_cpu():
    cores = os.cpu_count() or 4
    # Conservative thread setting for 16GB RAM laptops
    torch.set_num_threads(max(1, min(4, cores // 2)))
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

def load_tokenizer_model(local_only: bool):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        dtype=torch.float32,
        local_files_only=local_only,
        low_cpu_mem_usage=True
    )
    model.eval()
    return tok, model

def run(tok, model):
    msgs = [{"role": "user", "content": PROMPT}]
    enc = tok.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_new_tokens=40,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id
        )

    new_tokens = out[0][input_ids.shape[-1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

def main():
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Torch: {torch.__version__}")
    print(f"Model: {MODEL_ID}")
    tune_cpu()

    # 1) Try strict offline cache load first
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    tok = model = None
    cached = False
    try:
        tok, model = load_tokenizer_model(local_only=True)
        cached = True
        print("✅ Model found in local cache.")
    except Exception:
        print("ℹ️ Model not fully cached locally.")

    # 2) If not cached, do one-time online fetch
    if not cached:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)

        if not has_internet():
            print("❌ No internet and no local cache. Connect once and rerun.")
            return

        print("⬇️ Downloading model/tokenizer to cache...")
        tok, model = load_tokenizer_model(local_only=False)
        print("✅ Download complete. Next run can be offline.")

    # 3) Inference
    try:
        print("▶ Running inference...")
        text = run(tok, model)
        print("\n=== OUTPUT ===")
        print(text if text else "[empty output]")
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            print("❌ Inference failed due to OOM.")
            print("Try reducing threads and max_new_tokens (e.g., 24).")
        else:
            print("❌ Runtime error during inference:")
            print(e)
    except Exception:
        print("❌ Error during inference:")
        traceback.print_exc()

if __name__ == "__main__":
    main()