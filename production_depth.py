import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image
import time

class AdvancedDepthEstimator:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        
        print("[INFO] Loading Depth Anything V2 Model via HuggingFace...")
        # Automatically utilizes GPU if available
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            task="depth-estimation", 
            model="depth-anything/Depth-Anything-V2-Small-hf", 
            device=device
        )

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))

        frame_count = 0
        start_time = time.time()
        print("[INFO] Starting High-Fidelity Depth Processing...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert OpenCV BGR to PIL RGB for the HuggingFace Pipeline
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Run Inference
            depth_result = self.pipe(pil_img)
            
            # Extract depth array and resize to original video dimensions
            depth_array = np.array(depth_result["depth"])
            depth_resized = cv2.resize(depth_array, (width, height), interpolation=cv2.INTER_CUBIC)

            # Normalize to 0-255 for visualization
            depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Apply TURBO colormap (SOTA for depth clarity)
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
            
            # Stack and write
            combined = np.hstack((frame, depth_color))
            out.write(combined)
            
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                print(f"[INFO] Processed {frame_count} frames | {frame_count/elapsed:.2f} FPS")

        cap.release()
        out.release()
        print(f"✅ Success! Saved SOTA Depth Video to {self.output_path}")

if __name__ == "__main__":
    estimator = AdvancedDepthEstimator('Givenfiles/recording2.mp4', 'production_depth_output.mp4')
    estimator.process_video()