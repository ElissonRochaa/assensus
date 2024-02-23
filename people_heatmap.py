from ultralytics import YOLO
from ultralytics.solutions import heatmap
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("people.mp4")
assert cap.isOpened(), "Error: Cannot open video source"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


heatmap_obj = heatmap.Heatmap()
heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                        imw=cap.get(4),  # should same as im0 width
                        imh=cap.get(3),  # should same as im0 height
                        view_img=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        exit(0)
    results = model.track(im0, persist=True)
    frame = heatmap_obj.generate_heatmap(im0, tracks=results)