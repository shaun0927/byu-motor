"""Simple TensorBoard event file reader.

This module provides a helper function to print scalar metrics
logged during training or validation.  It uses TensorBoard's
``EventAccumulator`` to read ``events.out.tfevents.*`` files and
extract the values for the given tags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

__all__ = ["read_scalars", "print_scalars"]


def _find_event_file(logdir: str | Path) -> Path:
    logdir = Path(logdir)
    if logdir.is_file():
        return logdir
    events = sorted(logdir.glob("events.out.tfevents.*"))
    if not events:
        raise FileNotFoundError(f"no event file found in {logdir}")
    return events[-1]


def read_scalars(logdir: str | Path, tags: Iterable[str]) -> dict[str, list[tuple[int, float]]]:
    """Return scalar values for the specified tags.

    Parameters
    ----------
    logdir : str or Path
        Path to a directory containing an event file or the event file itself.
    tags : iterable of str
        Scalar tags to read.
    """
    event_file = _find_event_file(logdir)
    acc = EventAccumulator(str(event_file))
    acc.Reload()

    result: dict[str, list[tuple[int, float]]] = {}
    for tag in tags:
        result[tag] = [(e.step, e.value) for e in acc.Scalars(tag)]
    return result


def print_scalars(logdir: str | Path, tags: Iterable[str] | None = None) -> None:
    """Print scalar metrics stored in an event file.

    If *tags* is ``None``, ``{"val/f2", "val/tp", "val/fp", "val/fn"}`` is used.
    """
    if tags is None:
        tags = ["val/f2", "val/tp", "val/fp", "val/fn"]

    scalars = read_scalars(logdir, tags)
    header = "step\t" + "\t".join(tags)
    print(header)
    steps = [s[0] for s in scalars[next(iter(tags))]] if scalars else []
    for i, step in enumerate(steps):
        values = [f"{scalars[tag][i][1]:.6g}" for tag in tags]
        print(step, *values, sep="\t")


if __name__ == "__main__":  # pragma: no cover - simple CLI
    import argparse

    p = argparse.ArgumentParser(description="Print scalars from a TensorBoard event file")
    p.add_argument("logdir", type=str, help="Event file or directory containing it")
    p.add_argument("--tags", nargs="*", default=None, help="Scalar tags to display")
    args = p.parse_args()

    print_scalars(args.logdir, args.tags)
