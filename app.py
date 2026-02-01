import streamlit as st
import cv2
import os
import numpy as np
import csv
import pandas as pd
from datetime import datetime

# Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Paths
dataset_path = "captured_faces"
model_path = "face_model.yml"
label_map_path = "label_map.npy"

# ---------------------------
# Function: Capture New Student Data
# ---------------------------
def capture_student_data(student_name):
    output_dir = os.path.join(dataset_path, student_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            count += 1
            file_name = os.path.join(output_dir, f"{student_name}_{count}.jpg")
            cv2.imwrite(file_name, face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Captured: {count}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB")

        if count >= 100:
            st.success(f"✅ 100 images captured for {student_name}")
            break

    cap.release()

# ---------------------------
# Function: Retrain Model
# ---------------------------
def retrain_model():
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for user_name in os.listdir(dataset_path):
        user_folder = os.path.join(dataset_path, user_name)
        if not os.path.isdir(user_folder):
            continue

        label_map[current_label] = user_name

        for file in os.listdir(user_folder):
            img_path = os.path.join(user_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (200, 200))
            faces.append(img)
            labels.append(current_label)

        current_label += 1

    labels = np.array(labels)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)

    recognizer.save(model_path)
    np.save(label_map_path, label_map)

    st.success("✅ Model retrained successfully with new student data!")

# ---------------------------
# Function: Attendance System
# ---------------------------
def run_attendance(lecture_number, subject_name):
    csv_file = f"attendance_lecture{lecture_number}.csv"

    try:
        with open(csv_file, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time", "Lecture", "Subject"])
    except FileExistsError:
        pass

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    label_map = np.load(label_map_path, allow_pickle=True).item()

    cap = cv2.VideoCapture(0)
    marked_students = set()
    stframe = st.empty()
    stop_button = st.button("Stop Attendance")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label_id, confidence = recognizer.predict(roi)

            if confidence < 50:
                name = label_map[label_id]
            else:
                name = "Unknown"

            if name != "Unknown":
                if name not in marked_students:
                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")
                    time_str = now.strftime("%H:%M:%S")

                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([name, date_str, time_str, lecture_number, subject_name])

                    marked_students.add(name)
                    status = "Marked"
                else:
                    status = "Already Marked"
            else:
                status = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{name} - {status}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(rgb_frame, channels="RGB")

        if stop_button:
            break

    cap.release()

# ---------------------------
# Streamlit Dashboard
# ---------------------------
st.title("📖 Face Recognition Attendance Dashboard")

menu = st.sidebar.radio("Select Option", ["Add New Student", "Retrain Model", "Attendance System"])

if menu == "Add New Student":
    st.header("➕ Add New Student")
    student_name = st.text_input("Enter New Student Name")
    if st.button("Capture Data") and student_name:
        capture_student_data(student_name)

elif menu == "Retrain Model":
    st.header("🔄 Retrain Model")
    if st.button("Retrain Now"):
        retrain_model()

elif menu == "Attendance System":
    st.header("📝 Attendance System")
    lecture_number = st.text_input("Enter Lecture Number")
    subject_name = st.text_input("Enter Subject Name")

    if st.button("Start Attendance") and lecture_number and subject_name:
        run_attendance(lecture_number, subject_name)

    # ✅ Show Attendance Records + Download Option
    if st.button("Show Attendance Records") and lecture_number:
        csv_file = f"attendance_lecture{lecture_number}.csv"
        try:
            df = pd.read_csv(csv_file)
            st.table(df)
            st.download_button(
                label="Download Attendance CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=csv_file,
                mime="text/csv"
            )
        except FileNotFoundError:
            st.warning("No attendance file found for this lecture.")