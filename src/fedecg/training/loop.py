"""Epoch-level training with early stopping on validation macro AUROC.

The loop is written against plain `DataLoader`s and an `nn.Module` so that the
federated phase can reuse `train_one_epoch` and `predict` inside each Flower
client, and the DP phase can hand in an Opacus-wrapped model and optimizer.

Three choices from the PTB-XL benchmark (Strodthoff et al., 2021) are exposed in
the `training` config section, because they account for most of the gap
between a plain training loop and the published numbers:

**Random-crop training** (`crop_samples`). Each time a record is drawn, a
random window of `crop_samples` samples is cut from its 10 seconds. The model
sees a slightly different view of every record each epoch, which works as data
augmentation, and each forward pass is cheaper.

**Evaluation on the whole record.** The model ends in global pooling, so it
accepts any length: a model trained on crops can score all 10 seconds at once.
Optionally (`eval_window`, `eval_stride`, `eval_aggregate`) it scores
overlapping windows instead and averages (or max-pools) their probabilities,
as the benchmark does; on the validation fold that was no better and seven
times the compute, so the default is the whole record.

**Learning-rate schedule** (`lr_schedule`). `cosine` warms up linearly over the
first `warmup_fraction` of training, then decays to zero along a half cosine.
The factor is a function of training *progress* in [0, 1], so the same schedule
can be applied per optimizer step (centralized) or per round (federated).
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from fedecg.training.augment import augment_record, is_enabled
from fedecg.training.metrics import macro_auroc


def resolve_device(name: str = "auto") -> torch.device:
    """Map `auto | cpu | cuda | mps` to a `torch.device`.

    `auto` prefers CUDA, then Apple's MPS, then CPU.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unknown device {name!r}; use auto, cpu, cuda or mps")
    return torch.device(name)


class TrainingDataset(Dataset):
    """In-memory training records, randomly cropped and/or augmented.

    Each time a record is drawn it is cut to a random window of `length`
    samples (if given) and passed through `fedecg.training.augment` (if
    `augment` enables anything). Randomness comes from a seeded
    `torch.Generator` owned by the dataset, so the sequence of crops and
    augmentations depends only on the seed. That holds with `num_workers=0`;
    worker processes would each get a copy of the generator.
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        length: int | None = None,
        *,
        seed: int = 0,
        augment: Mapping[str, Any] | None = None,
        fs: float = 100.0,
    ):
        if length is not None and not 0 < length <= signals.shape[-1]:
            raise ValueError(f"crop length {length} must be in (0, {signals.shape[-1]}]")
        self.signals = torch.from_numpy(signals)
        self.labels = torch.from_numpy(labels)
        self.length = length
        self.augment = dict(augment) if is_enabled(augment) else None
        self.fs = fs
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[index]
        if self.length is not None:
            slack = signal.shape[-1] - self.length
            start = int(torch.randint(slack + 1, (1,), generator=self.generator))
            signal = signal[:, start : start + self.length]
        if self.augment is not None:
            signal = augment_record(signal, self.augment, self.generator, fs=self.fs)
        return signal, self.labels[index]


RandomCropDataset = TrainingDataset
"""Earlier name, kept so older code and notebooks still import."""


def make_loader(
    signals: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int = 0,
    num_workers: int = 0,
    crop_samples: int | None = None,
    augment: Mapping[str, Any] | None = None,
) -> DataLoader:
    """Wrap in-memory arrays in a `DataLoader`.

    Shuffling draws from its own seeded `torch.Generator`, so the batch order
    is reproducible and independent of any other use of torch's global RNG.
    With `crop_samples` and/or `augment` (the config's `training.augment`),
    every record is returned cropped and augmented: for training only;
    evaluation scores the whole, untouched record.
    """
    dataset: Dataset
    if crop_samples is None and not is_enabled(augment):
        dataset = TensorDataset(torch.from_numpy(signals), torch.from_numpy(labels))
    else:
        dataset = TrainingDataset(signals, labels, crop_samples, seed=seed, augment=augment)
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
    )


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017), averaged over records and classes.

    Scales each term of the cross-entropy by `(1 - p_t) ** gamma`, so records
    the model already gets right contribute little and hard ones (often the
    rare classes) dominate the gradient.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # noqa: D102
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        return ((1 - p_t) ** self.gamma * bce).mean()


LOSSES: tuple[str, ...] = ("bce", "weighted_bce", "focal")


def make_loss(training_config: Mapping[str, Any], labels: np.ndarray) -> nn.Module:
    """The training loss named by `training.loss` (default `bce`).

    Args:
        training_config: Uses `loss` and, for `focal`, `focal_gamma`.
        labels: Multi-hot training labels; `weighted_bce` weights each class's
            positives by its negative-to-positive ratio, so a class present in
            12% of records counts about as much as one present in 44%.
    """
    name = training_config.get("loss", "bce")
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        positives = labels.sum(axis=0)
        ratio = (len(labels) - positives) / np.maximum(positives, 1)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio, dtype=torch.float32))
    if name == "focal":
        return FocalLoss(float(training_config.get("focal_gamma", 2.0)))
    raise ValueError(f"Unknown loss {name!r}; use one of {LOSSES}")


def lr_factor(progress: float, schedule: str = "constant", warmup_fraction: float = 0.0) -> float:
    """Multiplier on the base learning rate at `progress` in [0, 1].

    Args:
        progress: Fraction of training completed.
        schedule: `constant` or `cosine` (linear warmup, then half-cosine decay).
        warmup_fraction: Share of training spent warming up (cosine only).
    """
    if schedule == "constant":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"Unknown lr_schedule {schedule!r}; use constant or cosine")
    progress = min(max(progress, 0.0), 1.0)
    if progress < warmup_fraction:
        return progress / warmup_fraction
    decay = (progress - warmup_fraction) / max(1.0 - warmup_fraction, 1e-12)
    return 0.5 * (1.0 + math.cos(math.pi * decay))


def proximal_term(model: nn.Module, anchor: Sequence[torch.Tensor]) -> torch.Tensor:
    """Squared L2 distance between the model's parameters and `anchor`.

    FedProx adds `mu / 2` times this to each client's loss, which keeps local
    training from drifting far from the global model it started at.
    """
    return sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor, strict=True))  # type: ignore[return-value]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    proximal_mu: float = 0.0,
    proximal_anchor: Sequence[torch.Tensor] | None = None,
) -> float:
    """One pass over `loader`. Returns the mean data loss per record.

    Args:
        model: Already on `device`.
        loader: Training batches.
        optimizer: Steps `model`'s parameters.
        loss_fn: Data loss, e.g. `BCEWithLogitsLoss`.
        device: Where `model` lives.
        scheduler: Stepped after every optimizer step.
        proximal_mu: FedProx coefficient. When positive, `proximal_anchor`
            (the global model the client received) must be given. The returned
            loss excludes the proximal term, so it stays comparable across
            algorithms.
        proximal_anchor: Parameters to stay close to, in `model.parameters()`
            order.
    """
    if proximal_mu > 0 and proximal_anchor is None:
        raise ValueError("proximal_mu > 0 needs a proximal_anchor")
    model.train()
    total, n_seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        objective = loss
        if proximal_mu > 0:
            objective = loss + proximal_mu / 2 * proximal_term(model, proximal_anchor)
        objective.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(x)
        n_seen += len(x)
    return total / max(n_seen, 1)


def _windows(x: torch.Tensor, window: int, stride: int) -> torch.Tensor:
    """`(n, leads, samples)` -> `(n, n_windows, leads, window)`, covering the end."""
    n_samples = x.shape[-1]
    starts = list(range(0, n_samples - window + 1, stride))
    if starts[-1] != n_samples - window:
        starts.append(n_samples - window)
    return torch.stack([x[..., s : s + window] for s in starts], dim=1)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    window: int | None = None,
    stride: int | None = None,
    aggregate: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Sigmoid probabilities and true labels for every record in `loader`.

    Args:
        model: Already on `device`.
        loader: Full-length records.
        device: Where `model` lives.
        window: If set, score overlapping windows of this many samples and
            combine them per record, as the model was trained on crops.
        stride: Step between windows. Defaults to half a window.
        aggregate: `mean` or `max` over the windows of a record.
    """
    if aggregate not in ("mean", "max"):
        raise ValueError(f"Unknown aggregate {aggregate!r}; use mean or max")
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        if window is None or window >= x.shape[-1]:
            p = torch.sigmoid(model(x))
        else:
            windows = _windows(x, window, stride or max(window // 2, 1))
            n, k = windows.shape[:2]
            p = torch.sigmoid(model(windows.flatten(0, 1))).view(n, k, -1)
            p = p.mean(dim=1) if aggregate == "mean" else p.amax(dim=1)
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def predict_from_config(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    training_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """`predict` with the evaluation windowing set in the `training` config."""
    return predict(
        model,
        loader,
        device,
        window=training_config.get("eval_window"),
        stride=training_config.get("eval_stride"),
        aggregate=training_config.get("eval_aggregate", "mean"),
    )


def bce_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary cross-entropy averaged over records and classes.

    Matches `BCEWithLogitsLoss` but works on probabilities, so it also applies
    to window-aggregated predictions.
    """
    p = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


@dataclass
class FitResult:
    """Outcome of `fit`. The model passed in is left holding the best weights."""

    history: list[dict[str, float]] = field(default_factory=list)
    """One dict per epoch: epoch, lr, train_loss, val_loss, val_macro_auroc, seconds."""
    best_epoch: int = 0
    best_score: float = -math.inf
    stopped_early: bool = False


EpochCallback = Callable[[dict[str, float]], None]

WrapFn = Callable[
    [nn.Module, torch.optim.Optimizer, DataLoader],
    tuple[nn.Module, torch.optim.Optimizer, DataLoader],
]
"""Replaces model, optimizer and loader before training, e.g. with DP versions."""


class EarlyStopping:
    """Track the best validation score and say when patience has run out.

    Shared by the centralized loop (per epoch) and the federated driver (per
    round), so both select models by exactly the same rule.
    """

    def __init__(self, patience: int):
        self.patience = patience
        self.best_score = -math.inf
        self.best_step = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self._since_best = 0

    def update(self, step: int, score: float, model: nn.Module) -> bool:
        """Record `score` at `step`. Returns True when training should stop."""
        # NaN (e.g. a validation set without positives) never counts as better.
        if score > self.best_score:
            self.best_score, self.best_step = score, step
            self.best_state = copy.deepcopy(model.state_dict())
            self._since_best = 0
            return False
        self._since_best += 1
        return self._since_best >= self.patience

    def restore(self, model: nn.Module) -> None:
        """Load the best weights seen so far into `model`."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config: Mapping[str, Any],
    device: torch.device,
    *,
    on_epoch: EpochCallback | None = None,
    progress: bool = False,
    wrap: WrapFn | None = None,
    loss_fn: nn.Module | None = None,
) -> FitResult:
    """Train with AdamW and stop when validation macro AUROC stops improving.

    Args:
        model: Already on `device`.
        train_loader: Shuffled training batches (random crops if the config
            sets `crop_samples`).
        val_loader: Full-length validation records, used for model selection.
        training_config: The config's `training` section (`epochs`,
            `learning_rate`, `weight_decay`, `early_stopping_patience`, and
            optionally `lr_schedule`, `warmup_fraction`, `crop_samples`,
            `eval_window`, `eval_stride`, `eval_aggregate`).
        device: Where `model` lives.
        on_epoch: Called with each epoch's metrics dict, e.g. to log to MLflow.
        progress: Print one line per epoch.
        loss_fn: Training loss (see `make_loss`); plain BCE by default.
        wrap: Applied to the model, optimizer and training loader before the
            first epoch (see `fedecg.privacy.PrivateTraining.wrap`). The
            wrapped model must share parameters with `model`, which is still
            the one evaluated and returned.

    Returns:
        A `FitResult`. On return, `model` holds the weights of the best epoch,
        not the last one.
    """
    metric = training_config.get("early_stopping_metric", "val_macro_auroc")
    if metric != "val_macro_auroc":
        raise ValueError(f"Unsupported early_stopping_metric {metric!r}")
    epochs = int(training_config["epochs"])
    stopper = EarlyStopping(int(training_config.get("early_stopping_patience", epochs)))

    loss_fn = (loss_fn or nn.BCEWithLogitsLoss()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    trained = model
    if wrap is not None:
        trained, optimizer, train_loader = wrap(model, optimizer, train_loader)
    schedule = training_config.get("lr_schedule", "constant")
    warmup = float(training_config.get("warmup_fraction", 0.0))
    total_steps = max(epochs * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step / total_steps, schedule, warmup)
    )

    result = FitResult()
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            trained, train_loader, optimizer, loss_fn, device, scheduler=scheduler
        )
        val_prob, val_true = predict_from_config(model, val_loader, device, training_config)
        score = macro_auroc(val_true, val_prob)

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": bce_from_probs(val_true, val_prob),
            "val_macro_auroc": score,
            "seconds": time.perf_counter() - start,
        }
        result.history.append(row)
        if on_epoch is not None:
            on_epoch(row)
        if progress:
            print(
                f"epoch {epoch:3d}  lr {lr:.2e}  train_loss {train_loss:.4f}  "
                f"val_loss {row['val_loss']:.4f}  val_macro_auroc {score:.4f}  "
                f"({row['seconds']:.1f}s)"
            )

        if stopper.update(epoch, score, model):
            result.stopped_early = epoch < epochs
            break

    result.best_epoch, result.best_score = stopper.best_step, stopper.best_score
    stopper.restore(model)
    return result
