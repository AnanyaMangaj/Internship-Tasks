import streamlit as st
import os
import pandas as pd

from backend.capture_faces import capture_student_faces
from backend.train_model import train_model

st.set_page_config(page_title="Smart Attendance System")

# Ensure folders exist
os.makedirs("dataset/student_images", exist_ok=True)
os.makedirs("attendance", exist_ok=True)

st.title("Smart Attendance System")

# -------- SESSION STATE --------

if "name" not in st.session_state:
    st.session_state.name = ""

if "usn" not in st.session_state:
    st.session_state.usn = ""

if "roll" not in st.session_state:
    st.session_state.roll = ""

if "student_class" not in st.session_state:
    st.session_state.student_class = ""

# -------- INPUTS --------

st.subheader("Student Registration")

name = st.text_input("Student Name", value=st.session_state.name)
usn = st.text_input("USN", value=st.session_state.usn)
roll = st.text_input("Roll Number", value=st.session_state.roll)
student_class = st.text_input("Class", value=st.session_state.student_class)

frame_placeholder = st.empty()
progress_bar = st.progress(0)
log_box = st.empty()

# -------- CAPTURE FACE --------

if st.button("Capture Faces"):

    if name.strip() == "" or usn.strip() == "":
        st.error("Enter student name and USN")
    else:

        st.session_state.name = name
        st.session_state.usn = usn
        st.session_state.roll = roll
        st.session_state.student_class = student_class

        log_box.text("Opening camera...")

        capture_student_faces(
            name,
            usn,
            frame_placeholder,
            log_box,
            progress_bar
        )

        st.success("50 images captured")

# -------- TRAIN MODEL --------

if st.button("Train Model"):

    log_box.text("Training model...")

    train_model(progress_bar, log_box)

    st.success("Model trained successfully")

# -------- MARK ATTENDANCE --------

if st.button("Start Attendance"):

    log_box.text("Starting face recognition...")

    os.system("python backend/recognize_face.py")

# -------- VIEW ATTENDANCE --------

if st.button("View Attendance"):

    file = "attendance/attendance.csv"

    if os.path.exists(file):

        df = pd.read_csv(file)

        st.dataframe(df)

    else:

        st.warning("No attendance records found")