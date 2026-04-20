"""Parser for Trekion IMU binary files.

The parser validates the file magic, discovers the most plausible record
layout from the binary data, and then streams records from disk.  The observed
record layout for the supplied file is:

    header/metadata: 64 bytes
    record:         <Q18f  (timestamp + 18 float32 values, 80 bytes)

The first ten float values are exposed as accel/gyro/mag/temp.  The remaining
values are retained as ``extra_*`` columns for auditability because they are
present in the proprietary record but are not required by the assignment.
"""

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

IMU_MAGIC = b"TRIMU001"

SENSOR_COLUMNS = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "temp",
]

EXTRA_COLUMNS = [f"extra_{idx}" for idx in range(8)]


@dataclass(frozen=True)
class ImuLayout:
    """Discovered IMU binary layout."""

    data_offset: int
    record_size: int
    struct_format: str
    value_count: int
    row_count: int
    score: float


def _validate_magic(handle) -> None:
    handle.seek(0)
    magic = handle.read(len(IMU_MAGIC))
    if magic != IMU_MAGIC:
        raise ValueError(f"Invalid IMU header: expected {IMU_MAGIC!r}, got {magic!r}")


def _candidate_offsets(file_size: int, record_size: int) -> Iterable[int]:
    upper = min(256, max(len(IMU_MAGIC), file_size - record_size))
    for offset in range(len(IMU_MAGIC), upper + 1, 4):
        if file_size >= offset + record_size and (file_size - offset) % record_size == 0:
            yield offset


def _values_are_sane(values: tuple[float, ...]) -> bool:
    if len(values) < len(SENSOR_COLUMNS):
        return False

    arr = np.asarray(values[: len(SENSOR_COLUMNS)], dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return False

    accel = arr[0:3]
    gyro = arr[3:6]
    temp = arr[9]

    return (
        np.all(np.abs(accel) <= 50.0)
        and np.all(np.abs(gyro) <= 1000.0)
        and -40.0 <= temp <= 125.0
    )


def _score_layout(path: Path, offset: int, record_struct: struct.Struct, row_count: int) -> float:
    timestamps: list[int] = []
    sane_values = 0
    checked = min(row_count, 512)

    with path.open("rb") as handle:
        handle.seek(offset)
        for _ in range(checked):
            raw = handle.read(record_struct.size)
            if len(raw) != record_struct.size:
                break
            unpacked = record_struct.unpack(raw)
            timestamp = int(unpacked[0])
            values = tuple(float(v) for v in unpacked[1:])
            timestamps.append(timestamp)
            if 0 < timestamp < 10**15 and _values_are_sane(values):
                sane_values += 1

    if len(timestamps) < 3:
        return -math.inf

    deltas = np.diff(np.asarray(timestamps, dtype=np.float64))
    positive_delta_ratio = float(np.mean(deltas > 0))
    finite_delta_ratio = float(np.mean(np.isfinite(deltas)))
    sane_ratio = sane_values / len(timestamps)

    # Prefer realistic high-rate IMU deltas after the basic validity checks.
    positive_deltas = deltas[deltas > 0]
    median_delta = float(np.median(positive_deltas)) if len(positive_deltas) else math.inf
    plausible_delta = 100_000.0 <= median_delta <= 10_000_000.0

    return (
        10.0 * sane_ratio
        + 8.0 * positive_delta_ratio
        + 2.0 * finite_delta_ratio
        + (3.0 if plausible_delta else 0.0)
    )


def discover_imu_layout(filepath: str | Path) -> ImuLayout:
    """Discover the IMU data offset and record structure.

    The supplied data has a clear ``<Q18f`` layout, but this function still
    scores plausible candidates instead of assuming an offset blindly.
    """

    path = Path(filepath)
    file_size = path.stat().st_size

    with path.open("rb") as handle:
        _validate_magic(handle)

    candidates: list[ImuLayout] = []
    for struct_format in ("<Q18f", "<Q10f"):
        record_struct = struct.Struct(struct_format)
        value_count = len(record_struct.unpack(b"\x00" * record_struct.size)) - 1
        for offset in _candidate_offsets(file_size, record_struct.size):
            row_count = (file_size - offset) // record_struct.size
            score = _score_layout(path, offset, record_struct, row_count)
            candidates.append(
                ImuLayout(
                    data_offset=offset,
                    record_size=record_struct.size,
                    struct_format=struct_format,
                    value_count=value_count,
                    row_count=row_count,
                    score=score,
                )
            )

    if not candidates:
        raise ValueError("Unable to discover a plausible IMU record layout")

    best = max(candidates, key=lambda item: item.score)
    if not math.isfinite(best.score) or best.score < 10.0:
        raise ValueError(f"No sane IMU layout found; best candidate was {best}")

    LOGGER.info(
        "IMU layout: offset=%s record_size=%s format=%s rows=%s score=%.2f",
        best.data_offset,
        best.record_size,
        best.struct_format,
        best.row_count,
        best.score,
    )
    return best


def parse_imu_file(filepath: str | Path, layout: ImuLayout | None = None) -> pd.DataFrame:
    """Parse a Trekion IMU file into a dataframe.

    Corrupted rows are skipped safely.  The file itself is streamed record by
    record; only parsed rows are accumulated for downstream vectorized sync.
    """

    path = Path(filepath)
    layout = layout or discover_imu_layout(path)
    record_struct = struct.Struct(layout.struct_format)

    rows: list[dict[str, float | int]] = []
    skipped = 0
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

            timestamp = int(unpacked[0])
            values = tuple(float(v) for v in unpacked[1:])

            if timestamp <= 0 or (last_timestamp is not None and timestamp <= last_timestamp):
                LOGGER.debug("Skipping non-monotonic IMU row %s timestamp=%s", row_index, timestamp)
                skipped += 1
                continue

            if not _values_are_sane(values):
                LOGGER.debug("Skipping out-of-range IMU row %s timestamp=%s", row_index, timestamp)
                skipped += 1
                continue

            last_timestamp = timestamp
            row: dict[str, float | int] = {"timestamp": timestamp}
            for name, value in zip(SENSOR_COLUMNS, values[: len(SENSOR_COLUMNS)]):
                row[name] = value
            for name, value in zip(EXTRA_COLUMNS, values[len(SENSOR_COLUMNS) :]):
                row[name] = value
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No valid IMU rows parsed from {path}")

    df.attrs["layout"] = layout
    df.attrs["skipped_rows"] = skipped

    LOGGER.info("Parsed %s IMU rows from %s; skipped=%s", len(df), path, skipped)
    return df
