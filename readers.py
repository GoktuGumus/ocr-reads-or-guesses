#!/usr/bin/env python3
"""The readers under test, ordered by how much language model sits in the loop.

    easyocr   CRNN + CTC. Decodes glyphs. Has no idea what a Turkish word is.
    trocr     Transformer encoder-decoder. A real text decoder, trained on lines.
    qwen-vl   A general vision-language model asked to transcribe.

That ordering is the experiment's axis. If prior-pull is what we think it is, it
should rise monotonically along it — and the cost of each step should show up as
errors that look like language rather than errors that look like noise.

Every reader returns a plain uppercase string with runs of whitespace collapsed,
because the comparison is about characters, not about who emits a trailing dot.
"""
from __future__ import annotations

import re

# The prompt is an experimental variable, not a detail. The first version of this
# benchmark used only `strict` and measured zero prior-pull — which may say more
# about that sentence than about the model. These three span the range from "no
# guidance" to "actively primed to expect Turkish words".
PROMPTS = {
    "neutral": "What text is in this image?",
    "strict": ("Transcribe the text in this image exactly as it appears, character "
               "by character. Output only the text, nothing else. Do not correct "
               "spelling and do not guess at unclear characters."),
    "primed": ("This is a Turkish road sign. Read the Turkish word on it and output "
               "only that word."),
}
PROMPT = PROMPTS["strict"]


def normalise(text: str) -> str:
    """Uppercase, collapse whitespace, drop surrounding punctuation.

    Turkish casing is a trap: str.upper() maps 'i' to 'I', not 'İ'. Getting that
    wrong would score a correct read as an error, so the two dotted/dotless pairs
    are mapped explicitly before the generic uppercase.
    """
    text = (text or "").replace("i", "İ").replace("ı", "I")
    text = text.upper().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;\"'`\n\t")


class EasyOCRReader:
    name = "easyocr"
    family = "ctc"

    def __init__(self, languages=("tr",), gpu: bool = True):
        import easyocr
        self.reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)

    def __call__(self, path: str) -> str:
        parts = self.reader.readtext(path, detail=0, paragraph=True)
        return normalise(" ".join(parts))


class TrOCRReader:
    name = "trocr"
    family = "encoder-decoder"

    def __init__(self, model_id: str = "microsoft/trocr-base-printed", device: str = "cuda"):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)
        self.device = device
        self.name = f"trocr:{model_id.split('/')[-1]}"

    def __call__(self, path: str) -> str:
        from PIL import Image
        image = Image.open(path).convert("RGB")
        pixels = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        ids = self.model.generate(pixels, max_new_tokens=32)
        return normalise(self.processor.batch_decode(ids, skip_special_tokens=True)[0])


class QwenVLReader:
    name = "qwen-vl"
    family = "vlm"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "cuda",
                 prompt: str = "strict"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device)
        self.prompt_style = prompt
        self.prompt = PROMPTS[prompt]
        self.name = f"qwen-vl:{model_id.split('/')[-1]}:{prompt}"

    def __call__(self, path: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": path}, {"type": "text", "text": self.prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        images, videos = process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos,
                                padding=True, return_tensors="pt").to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        return normalise(self.processor.batch_decode(trimmed, skip_special_tokens=True)[0])


READERS = {"easyocr": EasyOCRReader, "trocr": TrOCRReader, "qwen-vl": QwenVLReader}
