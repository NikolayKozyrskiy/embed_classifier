import os
import sys
from enum import Enum
from pathlib import Path
from shutil import copy
from typing import List, Optional

from matches.accelerators import DDPAccelerator
from matches.callbacks import (
    BestModelSaver,
    EnsureWorkdirCleanOrDevMode,
    LastModelSaverCallback,
    TqdmProgressCallback,
)
from matches.loop import Loop
from matches.utils import unique_logdir
import typer

from . import load_config, EClrConfig
from .common.wandb import WandBLoggingSink
from .train import train_ae_script, infer_ae_script


app = typer.Typer()


class DevMode(str, Enum):
    DISABLED = "disabled"
    SHORT = "short"
    OVERFIT_BATCH = "overfit-batch"


@app.command()
def train_ae(
    config_path: Path,
    comment: str = typer.Option(None, "--comment", "-C"),
    logdir: Path = None,
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    dev_mode: DevMode = typer.Option(DevMode.DISABLED, "--dev-mode", "-m"),
):
    config: EClrConfig = load_config(config_path, EClrConfig)

    comment = comment or config.comment
    if comment is None:
        comment = config_path.stem
    config.comment = comment

    logdir = logdir or unique_logdir(Path("logs/"), comment)

    callbacks = [
        WandBLoggingSink(comment, config),
        # EnsureWorkdirCleanOrDevMode(),
        # ProfilerCallback(),
        # MemLeakDebugCallback(),
        # MemProfilerCallback(),
        TqdmProgressCallback(),
    ]
    if dev_mode != DevMode.OVERFIT_BATCH:
        callbacks += [
            BestModelSaver(config.monitor),
            LastModelSaverCallback(),
        ]

    logdir.mkdir(exist_ok=True, parents=True)
    copy(config_path, logdir / "config.py", follow_symlinks=True)
    loop = Loop(
        logdir,
        callbacks,
        loader_override=dev_mode.value,
    )

    loop.launch(
        train_ae_script,
        DDPAccelerator(gpus),
        config=config,
    )


@app.command()
def infer_ae(
    config_path: Path,
    logdir: Path,
    checkpoint: str = typer.Option("best", "-c"),
    gpus: str = typer.Option(None, "--gpus", "--gpu", "-g"),
    data_root: Optional[Path] = None,
    output_name: Optional[str] = None,
):
    config = load_config(config_path, EClrConfig)
    loop = Loop(
        logdir,
        [TqdmProgressCallback()],
    )

    loop.launch(
        infer_ae_script,
        DDPAccelerator(gpus),
        config=config,
        checkpoint=checkpoint,
        data_root=data_root,
        output_name=output_name,
    )


if __name__ == "__main__":
    app()
