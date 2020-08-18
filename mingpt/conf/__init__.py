from dataclasses import dataclass
from typing import Any, Tuple, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class GPT1Conf:
    embd_pdrop: float = MISSING
    resid_pdrop: float = MISSING
    attn_pdrop: float = MISSING
    n_layer: int = MISSING
    n_head: int = MISSING
    n_embd: int = MISSING
    block_size: int = MISSING


@dataclass
class GPTModelTarget:
    _target_: str = "mingpt.model.GPT"
    config: GPT1Conf = MISSING


@dataclass
class CharDatasetConf:
    _target_: str = "mingpt.chardataset.CharDataset"
    url: str = MISSING
    filename: str = MISSING
    block_size: int = MISSING
    cache_dir: str = MISSING


@dataclass
class TrainerConfig:
    # optimization parameters
    max_epochs: int = MISSING
    batch_size: int = MISSING
    learning_rate: float = MISSING
    betas: Tuple[float, float] = MISSING
    grad_norm_clip: float = MISSING
    weight_decay: float = MISSING
    lr_decay: bool = MISSING
    warmup_tokens: int = MISSING
    final_tokens: int = MISSING
    ckpt_path: Optional[str] = MISSING
    num_workers: int = MISSING

@dataclass
class TrainerTarget:
    _target_: str = "mingpt.trainer.Trainer"
    config: TrainerConfig = MISSING


@dataclass
class Config:
    train_dataset: Any = MISSING
    test_dataset: Any = None
    model: Any = MISSING
    seed: int = MISSING
    trainer: Any = MISSING
    device: str = MISSING

    # checkpoint to load, interpreted as relative to cwd and not to the job working directory.
    load: Optional[str] = None

    # False to prevent training. useful when loading a checking for quick evaluation
    train: bool = True

    # evaluation input context
    context: str = MISSING


cs = ConfigStore.instance()
cs.store("config", node=Config)
cs.store(group="model", name="gpt1", node={"config": GPT1Conf})
cs.store(group="dataset", name="tinyshakespeare", node=CharDatasetConf)
cs.store(group="model", name="gpt1", node=GPTModelTarget)
cs.store(group="trainer", name="default", node=TrainerTarget)
