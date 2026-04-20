import cv2
import torch
import numpy as np

def run_depth_estimation(video_path, output_path, max_frames=50):
    print("Loading MiDaS Depth Estimation Model...")
    # Load the MiDaS Small model (fastest for CPU/basic GPU)
    model_type = "MiDaS_small" 
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()
    
    # Load transforms to resize images for the model
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # We want a side-by-side output, so the width is doubled
    out_width = width * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))

    frame_count = 0
    print("Starting frame processing...")

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Prepare image for MiDaS (Needs RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)

        # 2. Run Inference
        with torch.no_grad():
            prediction = midas(input_batch)
            
            # Resize the prediction to match the original video resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # 3. Format the Output
        depth_map = prediction.cpu().numpy()
        
        # Normalize the depth map to 0-255 for visualization
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        depth_8bit = (depth_normalized * 255.0).astype(np.uint8)
        
        # Apply the Inferno Colormap (as requested in the brief!)
        depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_INFERNO)
        
        # 4. Create Side-by-Side Frame
        combined_frame = np.hstack((frame, depth_color))
        
        # Write to output video
        out.write(combined_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"✅ Success! Saved side-by-side video to {output_path}")

if __name__ == "__main__":
    video_input = 'Givenfiles/recording2.mp4'
    video_output = 'depth_output_test.mp4'
    run_depth_estimation(video_input, video_output, max_frames=50)