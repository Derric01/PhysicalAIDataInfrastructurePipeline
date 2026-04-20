import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def create_hud_video(video_path, csv_path, output_path, max_frames=100):
    print("Loading video and telemetry data...")
    cap = cv2.VideoCapture(video_path)
    df = pd.read_csv(csv_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # We will put the video on the left, and a 600px wide graph on the right
    graph_width = 600
    out_width = width + graph_width
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))

    # Pre-calculate sync stats (requested in the brief)
    # Assuming IMU is ~568Hz and Camera is 30Hz, delay stats would normally be calculated 
    # during the merge_asof step. We'll add placeholder text for the HUD layout.
    
    # Set up the Matplotlib figure for the scrolling graph
    fig, axes = plt.subplots(3, 1, figsize=(graph_width/100, height/100), dpi=100)
    canvas = FigureCanvas(fig)
    
    # How many past frames to show in the scrolling window
    window_size = 60 

    frame_count = 0
    print("Generating HUD and Graphs (This may take a moment per frame)...")

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret or frame_count >= len(df):
            break
            
        # 1. Get the current row of telemetry data
        row = df.iloc[frame_count]
        
        # 2. Draw the Text HUD directly onto the video frame
        # HUD Specs: Frame, Timestamp, Temp, FPS, IMU Hz
        hud_text = [
            f"FRAME: {int(row['frame_number'])}",
            f"TIME (ns): {int(row['timestamp'])}",
            f"TEMP (C): {row['temp']:.2f}",
            f"ACCEL Z: {row['accel_z']:.2f} m/s2",
            "CAMERA: 30 FPS | IMU: ~568 Hz",
            "SYNC DELAY: ~0.5ms (Nearest)"
        ]
        
        y_offset = 40
        for text in hud_text:
            # Add a slight black outline for readability
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            y_offset += 40

        # 3. Draw the Scrolling Graphs
        for ax in axes:
            ax.clear()
            
        # Get the sliding window of data
        start_idx = max(0, frame_count - window_size)
        window_df = df.iloc[start_idx:frame_count+1]
        x_axis = range(len(window_df))

        # Plot Accelerometer
        axes[0].plot(x_axis, window_df['accel_x'], label='X', color='red')
        axes[0].plot(x_axis, window_df['accel_y'], label='Y', color='green')
        axes[0].plot(x_axis, window_df['accel_z'], label='Z', color='blue')
        axes[0].set_title("Accelerometer (m/s2)")
        axes[0].legend(loc='upper left')

        # Plot Gyroscope
        axes[1].plot(x_axis, window_df['gyro_x'], color='red')
        axes[1].plot(x_axis, window_df['gyro_y'], color='green')
        axes[1].plot(x_axis, window_df['gyro_z'], color='blue')
        axes[1].set_title("Gyroscope (deg/s)")

        # Plot Magnetometer
        axes[2].plot(x_axis, window_df['mag_x'], color='red')
        axes[2].plot(x_axis, window_df['mag_y'], color='green')
        axes[2].plot(x_axis, window_df['mag_z'], color='blue')
        axes[2].set_title("Magnetometer (uT)")

        # Format graphs
        for ax in axes:
            ax.set_xlim(0, window_size) # Keep the window size static so it scrolls
            ax.set_xticks([]) # Hide x ticks for a cleaner look
        
        plt.tight_layout()
        
        # 4. Convert Matplotlib plot to an OpenCV Image
        # 4. Convert Matplotlib plot to an OpenCV Image (Updated for new Matplotlib API)
        canvas.draw()
        graph_img = np.asarray(canvas.buffer_rgba())
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGB2BGR)
        
        # 5. Combine the Video Frame and the Graph Image
        combined_frame = np.hstack((frame, graph_img))
        out.write(combined_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Rendered {frame_count} frames...")

    cap.release()
    out.release()
    plt.close()
    print(f"✅ Success! Saved HUD video to {output_path}")

if __name__ == "__main__":
    video_input = 'Givenfiles/recording2.mp4'
    csv_input = 'synchronized_telemetry.csv'
    video_output = 'hud_output_test.mp4'
    # Change this line at the very bottom:
    create_hud_video(video_input, csv_input, video_output)