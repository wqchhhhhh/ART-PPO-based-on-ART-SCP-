"""Teacher-forced ART training with train/validation/test scenario separation."""

from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from cruise.config import MODEL_VERSION, Settings
from cruise.dataset import TrajectoryDataset
from cruise.model import ART, ModelConfig


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train(directory, output, config=ModelConfig(), epochs=50, batch_size=4,
          learning_rate=3e-5, accumulation=8, device="cpu", seed=42):
    seed_all(seed)
    train_data = TrajectoryDataset(directory, "train", seed, config.max_zones)
    val_data = TrajectoryDataset(directory, "val", seed, config.max_zones)
    if config.max_steps < train_data.settings.numerics.intervals:
        raise ValueError("Model timestep capacity is smaller than the training horizon")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    model = ART(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history, best = [], float("inf")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def batch_loss(batch, crop=False):
        b = {k: v.to(device) for k, v in batch.items()}
        length = b["states"].shape[1]
        start = random.randint(0, max(0, length-config.context_length)) if crop else 0
        windows = [(start, min(length, start+config.context_length))] if crop else [
            (i, min(length, i+config.context_length)) for i in range(0, length, config.context_length)]
        total, count = 0., 0
        for lo, hi in windows:
            inputs = {k: (v if k == "context" else v[:, lo:hi]) for k, v in b.items() if k != "next_states"}
            action, state = model(**inputs)
            loss = torch.mean((action-b["actions"][:, lo:hi])**2)+torch.mean((state-b["next_states"][:, lo:hi])**2)
            total = total+loss*(hi-lo)
            count += hi-lo
        return total/count

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count = 0., 0
        for i, batch in enumerate(train_loader):
            loss = batch_loss(batch, crop=True)
            # The last partial accumulation group uses its own denominator.
            group_start = (i//accumulation)*accumulation
            group_size = min(accumulation, len(train_loader)-group_start)
            (loss/group_size).backward()
            if (i+1) % accumulation == 0 or i+1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += loss.item()*len(batch["states"])
            count += len(batch["states"])
        model.eval()
        val_total, val_count = 0., 0
        with torch.no_grad():
            for batch in val_loader:
                val_total += batch_loss(batch).item()*len(batch["states"])
                val_count += len(batch["states"])
        row = {"epoch": epoch+1, "train_loss": total/count, "val_loss": val_total/val_count}
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_loss"] < best:
            best = row["val_loss"]
            torch.save({"model_version": MODEL_VERSION, "model_config": asdict(config),
                        "state_dict": model.state_dict(), "settings": train_data.settings.to_dict(),
                        "dataset_fingerprint": train_data.fingerprint,
                        "splits": {k: sorted(v) for k, v in train_data.group_splits.items()},
                        "epoch": epoch+1, "validation_loss": best,
                        "training": {"batch_size": batch_size, "accumulation": accumulation,
                                     "learning_rate": learning_rate, "seed": seed}}, output)
    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["model_version"] != MODEL_VERSION:
        raise ValueError("Checkpoint belongs to a different physical model")
    model = ART(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), Settings.from_dict(checkpoint["settings"]), checkpoint
