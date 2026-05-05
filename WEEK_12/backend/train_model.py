import face_recognition
import os
import pickle

DATASET_PATH = "dataset/student_images"
MODEL_PATH = "models/face_model.pkl"


def train_model(progress_bar=None, log_box=None):

    known_encodings = []
    known_names = []

    print("\nLoading dataset...\n")

    students = os.listdir(DATASET_PATH)

    for student in students:

        student_folder = os.path.join(DATASET_PATH, student)

        print(f"\nProcessing student: {student}")

        images = os.listdir(student_folder)

        for img_name in images:

            img_path = os.path.join(student_folder, img_name)

            print(f"Processing: {img_name}")

            image = face_recognition.load_image_file(img_path)

            face_locations = face_recognition.face_locations(image)

            face_encodings = face_recognition.face_encodings(image, face_locations)

            for encoding in face_encodings:

                known_encodings.append(encoding)
                known_names.append(student)

    print("\nSaving model...")

    data = {
        "encodings": known_encodings,
        "names": known_names
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(data, f)

    print("\nNew faces added:", student)
    print("Model updated successfully!\n")