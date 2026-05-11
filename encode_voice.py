from neutts import NeuTTS
import torch

# Use full PyTorch codec (NOT onnx) for encoding
tts = NeuTTS(
    backbone_repo="neuphonic/neutts-nano-q4-gguf",
    backbone_device="cpu",
    codec_repo="neuphonic/neucodec",   # <-- changed from neucodec-onnx-decoder
    codec_device="cpu"
)

ref_codes = tts.encode_reference("science_voice.wav")
torch.save(ref_codes, "my_voice_encoded_science.pt")
print("Voice encoded and saved!")