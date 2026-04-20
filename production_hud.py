import cv2
import pandas as pd
import numpy as np

def draw_vector_graph(canvas, data, color, x, y, w, h, title):
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (20, 20, 20), -1)
    cv2.putText(canvas, title, (x+10, y+25), 0, 0.6, (255, 255, 255), 1)
    if len(data) < 2: return
    
    # Auto-scaling logic
    d_min, d_max = np.min(data), np.max(data)
    norm = (data - d_min) / (d_max - d_min if d_max != d_min else 1)
    
    pts = np.column_stack((np.linspace(x, x+w, len(data)), (y+h-10) - (norm*(h-40)))).astype(np.int32)
    cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)

def run_hud_pipeline(video_path, csv_path, output_path):
    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    vw, vh, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (vw + 400, vh))

    for i in range(len(df)):
        ret, frame = cap.read()
        if not ret: break
        
        row = df.iloc[i]
        canvas = np.zeros((vh, vw + 400, 3), dtype=np.uint8)
        canvas[:, :vw] = frame
        
        # HUD Data [cite: 24]
        cv2.putText(canvas, f"FRAME: {int(row['frame_number'])}", (20, 40), 0, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, f"ACCEL Z: {row['acc_z']:.2f} m/s2", (20, 80), 0, 0.8, (0, 255, 0), 2)
        
        # Dynamic Graphs
        window = df.iloc[max(0, i-60):i+1]
        draw_vector_graph(canvas, window['acc_z'].values, (255, 100, 50), vw+10, 50, 380, 200, "Accel Z")
        draw_vector_graph(canvas, window['g_y'].values, (100, 255, 50), vw+10, 300, 380, 200, "Gyro Y")
        
        out.write(canvas)

    cap.release(); out.release()
    print(f"✅ HUD Video saved to {output_path}")

if __name__ == "__main__":
    run_hud_pipeline('Givenfiles/recording2.mp4', 'synchronized_telemetry.csv', 'hud_output.mp4')