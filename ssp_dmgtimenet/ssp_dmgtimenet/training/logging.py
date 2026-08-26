"""Lightweight logging adapters for the trainer.

We expose three loggers that the trainer can stack:

* :class:`RichLogger` writes coloured progress to stdout via ``rich``;
* :class:`TensorboardLogger` writes scalars to a TensorBoard run dir;
* :class:`MultiLogger` simply forwards every call to a list of loggers.

All loggers expose ``log_scalars``, ``log_text`` and ``close`` so the
trainer can call them uniformly.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Mapping, Sequence


class RichLogger:
    def __init__(self, name: str = "ssp-dmgtimenet") -> None:
        from rich.console import Console

        self.console = Console()
        self.name = name

    def log_scalars(self, prefix: str, scalars: Mapping[str, float], step: int) -> None:
        from rich.table import Table

        table = Table(title=f"[{self.name}] {prefix} step={step}", show_header=True)
        table.add_column("metric", justify="left")
        table.add_column("value", justify="right")
        for k in sorted(scalars):
            v = scalars[k]
            table.add_row(k, f"{v:.6f}" if isinstance(v, float) else str(v))
        self.console.print(table)

    def log_text(self, prefix: str, text: str, step: int) -> None:
        self.console.print(f"[{self.name}] [{prefix}@{step}] {text}")

    def close(self) -> None:
        return None


class TensorboardLogger:
    def __init__(self, log_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

    def log_scalars(self, prefix: str, scalars: Mapping[str, float], step: int) -> None:
        for k, v in scalars.items():
            self.writer.add_scalar(f"{prefix}/{k}", float(v), step)

    def log_text(self, prefix: str, text: str, step: int) -> None:
        self.writer.add_text(prefix, text, step)

    def close(self) -> None:
        with suppress(Exception):
            self.writer.flush()
        with suppress(Exception):
            self.writer.close()


class MultiLogger:
    def __init__(self, loggers: Sequence[object]) -> None:
        self._loggers = list(loggers)

    def log_scalars(self, prefix: str, scalars: Mapping[str, float], step: int) -> None:
        for logger in self._loggers:
            logger.log_scalars(prefix, scalars, step)

    def log_text(self, prefix: str, text: str, step: int) -> None:
        for logger in self._loggers:
            logger.log_text(prefix, text, step)

    def close(self) -> None:
        for logger in self._loggers:
            logger.close()
