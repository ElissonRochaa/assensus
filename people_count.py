from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("people.mp4")
assert cap.isOpened(), "Error: Cannot open video source"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


#region_points = [(20, 400), (950, 404), (950, 360), (20, 360)]
line_points = [(20, 420), (950, 420)]
classes_to_count = [0]

# Video writer
video_writer = cv2.VideoWriter("people_counting_output.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=line_points,
                 classes_names=model.names,
                 draw_tracks=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False, tracker="bytetrack.yaml",
                        classes=classes_to_count)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)


print(counter.counting_list)

cap.release()
video_writer.release()
cv2.destroyAllWindows()