"""Parser for Trekion video timestamp (VTS) binary files."""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

VTS_MAGIC = b"TRIVTS01"


@dataclass(frozen=True)
class VtsLayout:
    """Discovered VTS binary layout."""

    data_offset: int
    record_size: int
    struct_format: str
    row_count: int
    score: float
    has_aux_timestamp: bool


def _validate_magic(handle) -> None:
    handle.seek(0)
    magic = handle.read(len(VTS_MAGIC))
    if magic != VTS_MAGIC:
        raise ValueError(f"Invalid VTS header: expected {VTS_MAGIC!r}, got {magic!r}")


def _candidate_offsets(file_size: int, record_size: int) -> Iterable[int]:
    upper = min(256, max(len(VTS_MAGIC), file_size - record_size))
    for offset in range(len(VTS_MAGIC), upper + 1, 4):
        if file_size >= offset + record_size and (file_size - offset) % record_size == 0:
            yield offset


def _score_vts_layout(path: Path, offset: int, record_struct: struct.Struct, row_count: int) -> float:
    frames: list[int] = []
    timestamps: list[int] = []
    checked = min(row_count, 512)

    with path.open("rb") as handle:
        handle.seek(offset)
        for _ in range(checked):
            raw = handle.read(record_struct.size)
            if len(raw) != record_struct.size:
                break
            unpacked = record_struct.unpack(raw)
            frames.append(int(unpacked[0]))
            timestamps.append(int(unpacked[1]))

    if len(frames) < 3:
        return -math.inf

    frame_deltas = np.diff(np.asarray(frames, dtype=np.float64))
    timestamp_deltas = np.diff(np.asarray(timestamps, dtype=np.float64))
    positive_timestamp_ratio = float(np.mean(timestamp_deltas > 0))
    sequential_frame_ratio = float(np.mean(frame_deltas == 1))

    positive_deltas = timestamp_deltas[timestamp_deltas > 0]
    median_delta = float(np.median(positive_deltas)) if len(positive_deltas) else math.inf
    plausible_camera_delta = 10_000_000.0 <= median_delta <= 100_000_000.0

    starts_near_zero = frames[0] in (0, 1)
    return (
        10.0 * sequential_frame_ratio
        + 8.0 * positive_timestamp_ratio
        + (4.0 if plausible_camera_delta else 0.0)
        + (2.0 if starts_near_zero else 0.0)
    )


def discover_vts_layout(filepath: str | Path) -> VtsLayout:
    """Discover the VTS record layout from candidate structures."""

    path = Path(filepath)
    file_size = path.stat().st_size

    with path.open("rb") as handle:
        _validate_magic(handle)

    candidates: list[VtsLayout] = []
    for struct_format, has_aux in (("<IQIQ", True), ("<IQ", False)):
        record_struct = struct.Struct(struct_format)
        for offset in _candidate_offsets(file_size, record_struct.size):
            row_count = (file_size - offset) // record_struct.size
            score = _score_vts_layout(path, offset, record_struct, row_count)
            candidates.append(
                VtsLayout(
                    data_offset=offset,
                    record_size=record_struct.size,
                    struct_format=struct_format,
                    row_count=row_count,
                    score=score,
                    has_aux_timestamp=has_aux,
                )
            )

    if not candidates:
        raise ValueError("Unable to discover a plausible VTS record layout")

    best = max(candidates, key=lambda item: item.score)
    if not math.isfinite(best.score) or best.score < 12.0:
        raise ValueError(f"No sane VTS layout found; best candidate was {best}")

    LOGGER.info(
        "VTS layout: offset=%s record_size=%s format=%s rows=%s score=%.2f",
        best.data_offset,
        best.record_size,
        best.struct_format,
        best.row_count,
        best.score,
    )
    return best


def parse_vts_file(filepath: str | Path, layout: VtsLayout | None = None) -> pd.DataFrame:
    """Parse a Trekion VTS file into one row per video frame."""

    path = Path(filepath)
    layout = layout or discover_vts_layout(path)
    record_struct = struct.Struct(layout.struct_format)

    rows: list[dict[str, int]] = []
    skipped = 0
    last_frame: int | None = None
    last_timestamp: int | None = None

    with path.open("rb") as handle:
        _validate_magic(handle)
        handle.seek(layout.data_offset)

        for row_index in range(layout.row_count):
            raw = handle.read(layout.record_size)
            if len(raw) != layout.record_size:
                skipped += 1
                break

            try:
                unpacked = record_struct.unpack(raw[: record_struct.size])
            except struct.error:
                skipped += 1
                continue

            frame_number = int(unpacked[0])
            timestamp = int(unpacked[1])

            if (
                timestamp <= 0
                or (last_timestamp is not None and timestamp <= last_timestamp)
                or (last_frame is not None and frame_number <= last_frame)
            ):
                LOGGER.debug(
                    "Skipping non-monotonic VTS row %s frame=%s timestamp=%s",
                    row_index,
                    frame_number,
                    timestamp,
                )
                skipped += 1
                continue

            row = {"frame_number": frame_number, "timestamp": timestamp}
            if layout.has_aux_timestamp:
                row["aux_frame_number"] = int(unpacked[2])
                row["aux_timestamp"] = int(unpacked[3])

            last_frame = frame_number
            last_timestamp = timestamp
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No valid VTS rows parsed from {path}")

    frame_deltas = np.diff(df["frame_number"].to_numpy(dtype=np.int64))
    if len(frame_deltas) and not np.all(frame_deltas == 1):
        LOGGER.warning("VTS frames are monotonic but not strictly contiguous")

    df.attrs["layout"] = layout
    df.attrs["skipped_rows"] = skipped
    LOGGER.info("Parsed %s VTS rows from %s; skipped=%s", len(df), path, skipped)
    return df
