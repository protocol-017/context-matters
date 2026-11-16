# local hugging face inference and resumable output cache
# this module runs greedy four-bit generation and records raw batch outputs.

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

from snmpaper.RQ3.config import CACHE_DIR, MAX_NEW_TOKENS, MODELS, QUANTIZATION

class LLMRunner:
    def __init__(self, model_key: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.model_key = model_key
        self.model_id = MODELS[model_key]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=quant,
                                                          device_map="auto")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_path = CACHE_DIR / f"llm_{model_key}.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                self.cache[item["key"]] = item

    def _key(self, prompt: str) -> str:
        value = f"v1|{MAX_NEW_TOKENS}|{self.model_id}|nf4|{prompt}"
        return hashlib.sha1(value.encode()).hexdigest()

    def generate(self, prompts: list[str]) -> list[dict]:
        import torch
        results = []
        missing = []
        for prompt in prompts:
            key = self._key(prompt)
            if key in self.cache:
                results.append(self.cache[key])
            else:
                missing.append((key, prompt))
                results.append(None)
        for key, prompt in missing:
            started = time.perf_counter()
            rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                           tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            generated = output[0, inputs["input_ids"].shape[1]:]
            raw = self.tokenizer.decode(generated, skip_special_tokens=True).split(self.tokenizer.eos_token)[0]
            item = {"key": key, "raw": raw, "n_in": int(inputs["input_ids"].shape[1]),
                    "n_out": int(len(generated)), "seconds": time.perf_counter() - started}
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item) + "\n")
            self.cache[key] = item
        return [item for item in results if item is not None] if not missing else [self.cache[self._key(p)] for p in prompts]

    def meta(self) -> dict:
        import importlib.metadata
        return {"model_id": self.model_id, "revision": getattr(self.model.config, "_commit_hash", None),
                "tokenizer_class": self.tokenizer.__class__.__name__, "quantization": QUANTIZATION,
                "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
                "library_versions": {name: importlib.metadata.version(name) for name in ("transformers", "torch")},
                "hardware": platform.platform()}
