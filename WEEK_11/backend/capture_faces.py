import cv2
import os
import time


def capture_student_faces(name, usn, frame_placeholder, log_box, progress_bar):

    folder_name = f"{name}_{usn}"
    path = f"dataset/student_images/{folder_name}"

    os.makedirs(path, exist_ok=True)

    cap = cv2.VideoCapture(0)

    count = 0

    log_box.text("Starting webcam...")

    while count < 50:

        ret, frame = cap.read()

        if not ret:
            log_box.text("Camera error.")
            break

        img_path = f"{path}/{count}.jpg"
        cv2.imwrite(img_path, frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(frame)

        count += 1

        progress_bar.progress(count / 50)

        log_box.text(f"Capturing face images: {count}/50")

        time.sleep(0.05)

    cap.release()

    log_box.text("Face capture completed.")