from neutts import NeuTTS
import soundfile as sf
import torch
import numpy as np
import re
import sys
import os

# --- Configuration ---
INPUT_TEXT_FILE   = "input_text.txt"
VOICE_CODES_FILE  = "my_voice_encoded_dr_man.pt"
VOICE_REF_TEXT_FILE = "my_voice_drman.txt"
OUTPUT_FOLDER     = "clips"           # folder where all small clips are saved
SAMPLE_RATE       = 24000

def split_into_sentences(text):
    """
    Split full script into individual sentence/group clips.
    Rules:
      - Each non-empty line becomes its own clip.
      - Two sentences on the same line stay together as one clip.
      - Blank lines are ignored (they only create natural pause between clips).
    """
    lines = text.split("\n")
    sentences = []
    for line in lines:
        line = line.strip()
        if line:  # skip empty lines
            sentences.append(line)
    return sentences

def sanitize_filename(text, index):
    """Create a safe numbered filename from the sentence text."""
    short = re.sub(r'[^a-zA-Z0-9 ]', '', text[:40]).strip()
    short = re.sub(r'\s+', '_', short)
    return f"{index:03d}_{short}.wav"

def main():
    # --- Load input text ---
    if not os.path.exists(INPUT_TEXT_FILE):
        print(f"ERROR: '{INPUT_TEXT_FILE}' not found.")
        sys.exit(1)

    with open(INPUT_TEXT_FILE, "r", encoding="utf-8") as f:
        input_text = f.read().strip()

    if not input_text:
        print("ERROR: Input text file is empty.")
        sys.exit(1)

    print(f"Loaded input text ({len(input_text)} chars)")

    # --- Create output folder ---
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --- Load TTS model ---
    print("Loading NeuTTS model...")
    tts = NeuTTS(
        backbone_repo="neuphonic/neutts-nano-q4-gguf",
        backbone_device="cpu",
        codec_repo="neuphonic/neucodec-onnx-decoder",
        codec_device="cpu"
    )

    # --- Load voice clone ---
    if not os.path.exists(VOICE_CODES_FILE):
        print(f"ERROR: '{VOICE_CODES_FILE}' not found.")
        sys.exit(1)
    if not os.path.exists(VOICE_REF_TEXT_FILE):
        print(f"ERROR: '{VOICE_REF_TEXT_FILE}' not found.")
        sys.exit(1)

    ref_codes = torch.load(VOICE_CODES_FILE)
    ref_text  = open(VOICE_REF_TEXT_FILE, "r").read().strip()
    print("Voice clone loaded.")

    # --- Split text into sentence clips ---
    sentences = split_into_sentences(input_text)
    print(f"\nTotal clips to generate: {len(sentences)}\n")

    # --- Generate one clip per sentence ---
    generated_files = []
    for i, sentence in enumerate(sentences):
        filename  = sanitize_filename(sentence, i + 1)
        out_path  = os.path.join(OUTPUT_FOLDER, filename)

        # Skip if already generated (resume support)
        if os.path.exists(out_path):
            print(f"[{i+1}/{len(sentences)}] SKIP (exists): {filename}")
            generated_files.append(out_path)
            continue

        print(f"[{i+1}/{len(sentences)}] Generating: {sentence[:70]}")
        try:
            wav = tts.infer(sentence, ref_codes, ref_text)
            sf.write(out_path, wav, SAMPLE_RATE)
            duration = len(wav) / SAMPLE_RATE
            print(f"   Saved: {filename}  ({duration:.1f}s)")
            generated_files.append(out_path)
        except Exception as e:
            print(f"   ERROR on clip {i+1}: {e}")

    # --- Summary ---
    print(f"\n--- Done ---")
    print(f"Generated {len(generated_files)} clips in folder: ./{OUTPUT_FOLDER}/")
    print(f"\nAll clip filenames:")
    for f in generated_files:
        print(f"  {f}")
    print(f"\nPlay all clips in order:")
    print(f"  for f in {OUTPUT_FOLDER}/*.wav; do afplay \"$f\"; done")

if __name__ == "__main__":
    main()