import cv2
import pandas as pd
import numpy as np
import time

class VanguardHUD:
    def __init__(self, video_path, csv_path, output_path):
        self.cap = cv2.VideoCapture(video_path)
        self.df = pd.read_csv(csv_path)
        self.output_path = output_path
        
        # UI Constants
        self.panel_w = 450
        self.colors = {'x': (50, 50, 255), 'y': (50, 255, 50), 'z': (255, 150, 50), 'grid': (60, 60, 60)}
        self.font = cv2.FONT_HERSHEY_DUPLEX

    def draw_glass_panel(self, canvas, x, y, w, h):
        """Creates a modern semi-transparent 'glass' effect."""
        sub_img = canvas[y:y+h, x:x+w]
        rect = np.zeros(sub_img.shape, dtype=np.uint8)
        cv2.rectangle(rect, (0, 0), (w, h), (20, 20, 20), -1)
        canvas[y:y+h, x:x+w] = cv2.addWeighted(sub_img, 0.5, rect, 0.5, 0)
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (100, 100, 100), 1, cv2.LINE_AA)

    def draw_horizon(self, canvas, x_c, y_c, radius, acc_x, acc_y, acc_z):
        """Renders a Virtual Horizon indicator using Accelerometer data."""
        # Calculate Pitch/Roll from gravity vector
        roll = np.arctan2(acc_y, acc_z)
        pitch = np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2))
        
        # Draw Gauge Base
        cv2.circle(canvas, (x_c, y_c), radius, (50, 50, 50), -1)
        cv2.circle(canvas, (x_c, y_c), radius, (200, 200, 200), 2, cv2.LINE_AA)
        
        # Calculate Horizon Line points
        p1 = (int(x_c - radius * np.cos(roll)), int(y_c - radius * np.sin(roll) + pitch * 50))
        p2 = (int(x_c + radius * np.cos(roll)), int(y_c + radius * np.sin(roll) + pitch * 50))
        cv2.line(canvas, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "HORIZON", (x_c-30, y_c+radius+20), self.font, 0.5, (200, 200, 200), 1)

    def draw_multi_graph(self, canvas, data_dict, x, y, w, h, title):
        """Draws professional fused-axis graphs with grids and legends."""
        self.draw_glass_panel(canvas, x, y, w, h)
        cv2.putText(canvas, title, (x+10, y+25), self.font, 0.6, (255, 255, 255), 1)
        
        # Draw Grid Lines
        for i in range(1, 4):
            gy = y + (h // 4) * i
            cv2.line(canvas, (x+5, gy), (x+w-5, gy), self.colors['grid'], 1)

        # Plot Axes (X, Y, Z)
        for axis, values in data_dict.items():
            if len(values) < 2: continue
            # Auto-scale and smooth using simple moving average
            smoothed = pd.Series(values).rolling(window=5, min_periods=1).mean().values
            d_min, d_max = np.min(smoothed), np.max(smoothed)
            norm = (smoothed - d_min) / (d_max - d_min if d_max != d_min else 1)
            
            pts = np.column_stack((
                np.linspace(x+5, x+w-5, len(smoothed)), 
                (y+h-10) - (norm * (h-40))
            )).astype(np.int32)
            cv2.polylines(canvas, [pts], False, self.colors[axis], 2, cv2.LINE_AA)
            
            # Tiny Legend
            lx = x + 10 + (40 if axis=='y' else 80 if axis=='z' else 0)
            cv2.putText(canvas, axis.upper(), (lx, y+h-5), self.font, 0.4, self.colors[axis], 1)

    def render(self):
        vw, vh = int(self.cap.get(3)), int(self.cap.get(4))
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (vw + self.panel_w, vh))

        for i in range(len(self.df)):
            ret, frame = self.cap.read()
            if not ret: break
            
            row = self.df.iloc[i]
            # Master Canvas: Video + HUD Side Panel
            canvas = np.zeros((vh, vw + self.panel_w, 3), dtype=np.uint8)
            canvas[:, :vw] = frame
            
            # --- Sidebar Content ---
            win = self.df.iloc[max(0, i-80):i+1] # Rolling window
            
            # 1. Multi-Axis Accelerometer
            acc_data = {'x': win['acc_x'].values, 'y': win['acc_y'].values, 'z': win['acc_z'].values}
            self.draw_multi_graph(canvas, acc_data, vw+10, 20, 430, 220, "Linear Acceleration (m/s2)")
            
            # 2. Virtual Horizon (Spatial Awareness)
            self.draw_horizon(canvas, vw + 225, 380, 80, row['acc_x'], row['acc_y'], row['acc_z'])
            
            # 3. Dynamic Telemetry Text
            cv2.putText(canvas, f"STATUS: TRACKING", (vw+20, vh-60), self.font, 0.7, (0, 255, 0), 2)
            cv2.putText(canvas, f"TEMP: {row['temp']:.1f}C", (vw+20, vh-30), self.font, 0.7, (255, 255, 255), 1)

            out.write(canvas)
            if i % 100 == 0: print(f"Processing... {i}/{len(self.df)}")

        self.cap.release(); out.release()
        print(f"✅ Vanguard HUD complete: {self.output_path}")

if __name__ == "__main__":
    hud = VanguardHUD('Givenfiles/recording2.mp4', 'synchronized_telemetry.csv', 'vanguard_hud_final.mp4')
    hud.render()