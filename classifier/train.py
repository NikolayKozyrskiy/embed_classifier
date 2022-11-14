from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
import warnings

import torch
import torch.nn as nn
import ignite.distributed as idist
from ignite.distributed.auto import auto_dataloader
from ignite.metrics.accumulation import Average

from matches.loop import Loop
from matches.loop.loader_scheduling import DataloaderSchedulerWrapper
from matches.shortcuts.optimizer import SchedulerScopeType
from matches.utils import seed_everything, setup_cudnn_reproducibility

from .config import EClrConfig
from .common.utils import enumerate_normalized, log_optimizer_lrs

warnings.filterwarnings("ignore", module="torch.optim.lr_scheduler")


def train_script(loop: Loop, config: EClrConfig):
    seed_everything(42)
    setup_cudnn_reproducibility(False, True)


def infer_script(
    loop: Loop,
    config: EClrConfig,
    checkpoint: str = "best",
    data_root: Optional[Path] = None,
    folds: Optional[List[str]] = (),
    split: Optional[str] = None,
    output_name: Optional[str] = None,
):
    device = f"cuda:{torch.cuda.current_device()}"
