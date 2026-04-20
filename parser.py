import struct
import binascii

def hunt_data_structure(filepath, file_type):
    print(f"\n--- Hunting inside {file_type} ---")
    with open(filepath, 'rb') as f:
        content = f.read()

    # The exact bytes for timestamp 34577390724 (which we found in both files)
    target_timestamp = struct.pack('<Q', 34577390724)
    
    # Find exactly where this timestamp lives in the file
    idx = content.find(target_timestamp)
    
    if idx == -1:
        print("Could not find the target timestamp.")
        return

    print(f"SUCCESS: Found timestamp starting exactly at byte offset: {idx}")

    if file_type == 'VTS':
        # VTS should be Frame + Timestamp. Let's look at the 8 bytes BEFORE the timestamp
        # and 8 bytes AFTER the timestamp to see where the frame number (0, 1, 2...) is hiding.
        before = content[idx-8:idx]
        after = content[idx+8:idx+16]
        
        print("Bytes BEFORE timestamp (Could be frame number):", [struct.unpack('<I', before[i:i+4])[0] for i in range(0, 8, 4)])
        print("Bytes AFTER timestamp (Could be next frame):", [struct.unpack('<I', after[i:i+4])[0] for i in range(0, 8, 4)])

    elif file_type == 'IMU':
        # For IMU, let's unpack the 48 bytes immediately AFTER the timestamp as floats.
        # We are looking for a clear 9.8 (gravity) and ~25.0 (temperature) in this list.
        data_after = content[idx+8 : idx+8+48]
        
        floats = []
        for i in range(0, len(data_after), 4):
            # Unpack 4 bytes into a float
            try:
                val = struct.unpack('<f', data_after[i:i+4])[0]
                floats.append(round(val, 4)) # Round for easy reading
            except:
                pass
                
        print(f"Floats immediately following the timestamp:\n{floats}")
        print("\nCan you spot Earth's gravity (~9.8) or a temperature (~20-30) in this list?")

# --- Execution ---
if __name__ == "__main__":
    hunt_data_structure('Givenfiles/recording2.vts', 'VTS')
    hunt_data_structure('Givenfiles/recording2.imu', 'IMU')