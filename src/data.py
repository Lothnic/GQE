# src/data.py
# §4.1 — FineWeb-Edu Streaming Dataloader and Packing

import time
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Iterator, Dict, Any, Optional


class PackedTokenDataset(IterableDataset):
    """§4.1 — Dataset that streams FineWeb-Edu and packs tokens into max_seq_len chunks.
    
    Concatenates tokenized text documents and splits them into fixed-size chunks
    of length max_seq_len to avoid padding waste during pretraining.
    """
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        max_seq_len: int,
        split: str = "train",
        streaming: bool = True,
        max_stream_retries: int = 20,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.split = split
        self.streaming = streaming
        self.max_stream_retries = max_stream_retries
        
        # Load tokenizer (raise model_max_length so long docs don't warn)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.model_max_length = int(1e9)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.eos_token_id = self.tokenizer.eos_token_id

    def _open_dataset(self):
        return load_dataset(
            self.dataset_name,
            name="sample-10BT" if "fineweb-edu" in self.dataset_name.lower() else None,
            split=self.split,
            streaming=self.streaming,
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # §4.1 — FineWeb-Edu stream with reconnect on transient HF/network errors
        buffer: list = []
        target_len = self.max_seq_len + 1
        retries = 0

        while True:
            try:
                dataset = self._open_dataset()
                for sample in dataset:
                    retries = 0  # reset after successful sample
                    text = sample.get("text", "")
                    if not text:
                        continue

                    tokens = self.tokenizer.encode(
                        text, add_special_tokens=False, truncation=False
                    )
                    tokens.append(self.eos_token_id)
                    buffer.extend(tokens)

                    while len(buffer) >= target_len:
                        chunk = buffer[:target_len]
                        buffer = buffer[self.max_seq_len :]
                        yield {
                            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                            "labels": torch.tensor(chunk[1:], dtype=torch.long),
                        }
                # Stream exhausted cleanly — stop
                return
            except Exception as e:
                retries += 1
                if retries > self.max_stream_retries:
                    raise RuntimeError(
                        f"FineWeb stream failed after {self.max_stream_retries} retries: {e}"
                    ) from e
                wait = min(60.0, 2.0 ** min(retries, 5))
                print(
                    f"[data] stream error ({type(e).__name__}: {e}); "
                    f"retry {retries}/{self.max_stream_retries} in {wait:.0f}s"
                )
                time.sleep(wait)

def get_dataloader(config: Dict[str, Any], batch_size: int, split: str = "train") -> DataLoader:
    """Returns a PyTorch DataLoader wrapping the PackedTokenDataset."""
    dataset_cfg = config["data"]
    model_cfg = config["model"]
    
    tokenizer_name = dataset_cfg.get("tokenizer", "gpt2")
    dataset_name = dataset_cfg.get("dataset", "HuggingFaceFW/fineweb-edu")
    
    dataset = PackedTokenDataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_seq_len=model_cfg["max_seq_len"],
        split=split,
        streaming=dataset_cfg.get("streaming", True)
    )
    
    # We do not use collate_fn since dataset already yields tensors of exact shape
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=dataset_cfg.get("num_workers", 0),
        pin_memory=True
    )
