import struct
import pandas as pd

def parse_vts(filepath):
    print(f"Parsing VTS: {filepath}...")
    vts_data = []
    
    # Final Schema: Start at 32, <IQ (4-byte Frame, 8-byte Timestamp)
    start_offset = 32
    row_format = '<IQ'
    row_size = struct.calcsize(row_format)

    with open(filepath, 'rb') as f:
        content = f.read()

    for i in range(start_offset, len(content) - row_size + 1, row_size):
        row_bytes = content[i:i + row_size]
        try:
            frame, timestamp = struct.unpack(row_format, row_bytes)
            vts_data.append((frame, timestamp))
        except struct.error:
            break

    df = pd.DataFrame(vts_data, columns=['frame_number', 'timestamp'])
    print(f"✅ Success: Parsed {len(df)} video frames.")
    return df

def parse_imu(filepath):
    print(f"Parsing IMU: {filepath}...")
    imu_data = []
    
    # Final Schema: Start at 28, <Q10f (8-byte Timestamp, 10 Floats)
    start_offset = 28
    row_format = '<Q10f'
    row_size = struct.calcsize(row_format)

    with open(filepath, 'rb') as f:
        content = f.read()

    for i in range(start_offset, len(content) - row_size + 1, row_size):
        row_bytes = content[i:i + row_size]
        try:
            unpacked = struct.unpack(row_format, row_bytes)
            imu_data.append(unpacked)
        except struct.error:
            break

    columns = [
        'timestamp',
        'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'mag_x', 'mag_y', 'mag_z',
        'temp'
    ]
    df = pd.DataFrame(imu_data, columns=columns)
    print(f"✅ Success: Parsed {len(df)} IMU readings.")
    return df

if __name__ == "__main__":
    # 1. Parse the data
    vts_df = parse_vts('Givenfiles/recording2.vts')
    imu_df = parse_imu('Givenfiles/recording2.imu')
    
    # 2. Prepare for Synchronization
    print("\n--- Starting Synchronization ---")
    
    # FIX: Force Pandas to treat both timestamps as exactly the same data type
    vts_df['timestamp'] = vts_df['timestamp'].astype('uint64')
    imu_df['timestamp'] = imu_df['timestamp'].astype('uint64')
    
    # merge_asof requires both dataframes to be strictly sorted by the timestamp
    vts_df = vts_df.sort_values('timestamp')
    imu_df = imu_df.sort_values('timestamp')

    # 3. Perform the As-Of Merge
    # This says: "For every video frame in vts_df, find the IMU row 
    # whose timestamp is closest to it."
    sync_df = pd.merge_asof(
        vts_df, 
        imu_df, 
        on='timestamp', 
        direction='nearest'
    )
    
    print(f"✅ Success: Created synchronized dataset with {len(sync_df)} rows.")
    
    # 4. Save to CSV so we don't have to parse the binary files ever again!
    output_filename = 'synchronized_telemetry.csv'
    sync_df.to_csv(output_filename, index=False)
    print(f"💾 Saved synchronized data to {output_filename}")
    
    # Let's peek at the first few synchronized rows
    print("\n--- Synchronized Data (First 5 Frames) ---")
    print(sync_df[['frame_number', 'timestamp', 'accel_z', 'temp']].head())