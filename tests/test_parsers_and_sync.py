from pathlib import Path

import numpy as np

from parsers.imu_parser import SENSOR_COLUMNS, discover_imu_layout, parse_imu_file
from parsers.vts_parser import discover_vts_layout, parse_vts_file
from sync.synchronizer import synchronize_streams
from utils.validation import normalize_timestamps

ROOT = Path(__file__).resolve().parents[1]
IMU_PATH = ROOT / "Givenfiles" / "recording2.imu"
VTS_PATH = ROOT / "Givenfiles" / "recording2.vts"


def test_imu_parser_discovers_valid_layout():
    layout = discover_imu_layout(IMU_PATH)
    assert layout.data_offset == 64
    assert layout.record_size == 80
    assert layout.struct_format == "<Q18f"

    imu_df = parse_imu_file(IMU_PATH, layout)
    assert len(imu_df) == 24938
    assert np.all(np.diff(imu_df["timestamp"].to_numpy(dtype=np.int64)) > 0)
    assert imu_df[["accel_x", "accel_y", "accel_z"]].abs().max().max() < 50
    assert imu_df[["gyro_x", "gyro_y", "gyro_z"]].abs().max().max() < 1000


def test_vts_parser_discovers_valid_layout():
    layout = discover_vts_layout(VTS_PATH)
    assert layout.data_offset == 32
    assert layout.record_size == 24
    assert layout.struct_format == "<IQIQ"

    vts_df = parse_vts_file(VTS_PATH, layout)
    assert len(vts_df) == 1316
    assert vts_df["frame_number"].iloc[0] == 0
    assert np.all(np.diff(vts_df["frame_number"].to_numpy(dtype=np.int64)) == 1)
    assert np.all(np.diff(vts_df["timestamp"].to_numpy(dtype=np.int64)) > 0)


def test_synchronization_interpolates_one_row_per_frame():
    imu_df, _ = normalize_timestamps(parse_imu_file(IMU_PATH), stream_name="IMU")
    vts_df, _ = normalize_timestamps(parse_vts_file(VTS_PATH), stream_name="VTS")

    synced_df, metrics = synchronize_streams(vts_df, imu_df)

    assert len(synced_df) == 1316
    assert metrics["sync_method"] == "linear_interpolation"
    assert metrics["max_delay_ms"] < 1.0
    assert metrics["imu_rate_hz"] > 560
    assert metrics["imu_rate_hz"] < 575
    assert np.isfinite(synced_df[SENSOR_COLUMNS].to_numpy()).all()
