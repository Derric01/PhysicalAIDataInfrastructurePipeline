import cv2
from ultralytics import YOLOWorld

def run_advanced_segmentation(video_path, output_path):
    # Load Open-Vocabulary model
    model = YOLOWorld('yolov8s-world.pt') 
    # Set custom classes for THIS specific scene 
    model.set_classes(["calibration checkerboard", "soldering iron", "computer mouse", "hand"])

    cap = cv2.VideoCapture(video_path)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = model.predict(frame, conf=0.2, verbose=False)
        out.write(results[0].plot())

    cap.release(); out.release()
    print(f"✅ Segmentation Video saved to {output_path}")

if __name__ == "__main__":
    run_advanced_segmentation('Givenfiles/recording2.mp4', 'segmentation_output.mp4')