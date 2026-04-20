import cv2
from ultralytics import YOLO

def run_segmentation(video_path, output_path):
    print("Loading YOLOv8 Segmentation Model...")
    # This will automatically download the lightweight 'nano' segmentation model
    model = YOLO('yolov8n-seg.pt') 
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    print("Starting segmentation processing...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Run YOLOv8 Inference
        # conf=0.25 filters out weak, unconfident detections
        results = model(frame, conf=0.25, verbose=False)
        
        # 2. Draw the overlays
        # The .plot() method automatically draws bounding boxes, labels, scores, and masks!
        annotated_frame = results[0].plot()

        # 3. Write to output video
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"✅ Success! Saved segmentation video to {output_path}")

if __name__ == "__main__":
    video_input = 'Givenfiles/recording2.mp4'
    video_output = 'segmentation_output.mp4'
    run_segmentation(video_input, video_output)