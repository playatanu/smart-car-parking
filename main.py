from ultralytics import YOLO
import cv2
import time
from datetime import datetime
import easyocr
import csv

car_detector = YOLO("models/yolo11n.pt")
license_plate_detector = YOLO("models/license_plate_detector.pt")

reader = easyocr.Reader(["en"], gpu=False)

class_list = car_detector.names

window_h = 720
window_w = 1024
line_y = 300
line_x = window_w // 2 - 100

car_exit = set()
counter = 0
crossed = {}

frame_idx = 0  # Track frame number
prev_time = time.time()  # Initialize time for FPS calculation

csv_file = f"output/{prev_time}.csv"

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_url", "license", "timestamp"])

cap = cv2.VideoCapture("videos/cctv_cam.mp4")
# cap = cv2.VideoCapture(1)

cv2.namedWindow("car croped window", cv2.WINDOW_NORMAL)  # Allow manual resizing
cv2.resizeWindow("car croped window", 400, 400)  # Set fixed size

while cap.isOpened():
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not ret:
        break

    # resize the frame
    frame = cv2.resize(frame, (window_w, window_h))

    # get predction results from frame
    results = car_detector.track(frame, persist=True, verbose=False)

    # store results
    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

    # extract one by one objects
    for box, track_id, class_idx, conf in zip(
        boxes, track_ids, class_indices, confidences
    ):
        # bbox x1y1, x2,y2
        x1, y1, x2, y2 = map(int, box)
        # center of the bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # detect if its car or van
        if x1 > line_x and (class_idx == 2 or class_idx == 7):

            # draw center point
            cv2.circle(frame, (center_x, center_y), 4, (0, 255, 255), -1)

            # put text [track_id]
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                1,
            )

            # add boundary box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

            if line_y - 20 <= center_y <= line_y + 20:

                if track_id not in crossed:
                    crossed[track_id] = True
                    counter = counter + 1

                    car_image = frame[y1:y2, x1:x2]

                    # license plate detection
                    license_plates = license_plate_detector(car_image, verbose=False)[0]
                    for license_plate in license_plates.boxes.data.tolist():
                        x_1, y_1, x_2, y_2, score, class_id = license_plate
                        x_1, y_1, x_2, y_2 = int(x_1), int(y_1), int(x_2), int(y_2)

                        cv2.rectangle(
                            car_image, (x_1, y_1), (x_2, y_2), (0, 255, 255), 1
                        )
                        license_plate_image = car_image[y_1:y_2, x_1:x_2]

                        # image to text [OCR]
                        detections = reader.readtext(license_plate_image)
                        _, license_plate_number_text, _ = detections[0]

                        cv2.putText(
                            car_image,
                            f"{license_plate_number_text}",
                            (x_1, y_1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

                    current_time = datetime.now().strftime("%H:%M:%S %p")

                    cv2.putText(
                        car_image,
                        f"{current_time}",
                        (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    cv2.imshow("car croped window", car_image)
                    with open(csv_file, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [
                                f"car_{counter}.jpg",
                                license_plate_number_text,
                                current_time,
                            ]
                        )

    # Defalut UI

    # Calculate FPS
    curr_time = time.time()
    elapsed_time = curr_time - prev_time
    current_fps = 1 / elapsed_time if elapsed_time > 0 else 0
    prev_time = curr_time  # Update previous time

    # Car counter text
    cv2.putText(
        frame,
        f"CAR Exit: {counter}",
        (30, window_h - 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"fps {current_fps:.4}",
        (30, window_h - 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.line(frame, (line_x, line_y), (window_w, line_y), (0, 0, 255), 1)
    cv2.imshow("window", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
