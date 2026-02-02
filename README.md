Live link : https://ai-attendance-system-y3wz.onrender.com
# 🎓 Face Recognition Attendance Dashboard

A Streamlit-based web application for managing student attendance using **Face Recognition**.  
This project allows you to:
- ➕ Add new student data (capture 100 images via webcam)
- 🔄 Retrain the recognition model (LBPH)
- 📝 Mark attendance lecture-wise
- 📊 View attendance records
- 📥 Download attendance CSV files

---

## 🚀 Features
- **Face Detection & Recognition** using OpenCV Haar Cascade + LBPH.
- **Add New Student** directly from the web interface.
- **Retrain Model** with updated dataset.
- **Lecture-wise Attendance** stored in separate CSV files.
- **Duplicate Prevention** (each student marked once per lecture).
- **Download Attendance CSV** from the dashboard.
---

## 🛠️ Tech Stack
- [Python 3.9+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)

---


## 📂 Project Structure


├── app.py                # Main Streamlit application ├── requirements.txt      # Dependencies ├── captured_faces/       # Dataset folder (student images) ├── face_model.yml        # Trained LBPH model ├── label_map.npy         # Label mapping (ID → Name) └── attendance_lectureX.csv  # Attendance records per lecture

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/face-attendance-dashboard.git
   cd face-attendance-dashboard


- Install dependencies
pip install -r requirements.txt
- Run the app
streamlit run app.py


- Open in browser:
http://localhost:8501



🌐 Deployment on Render
- Push your repo to GitHub.
- On Render, create a New Web Service.
- Connect your GitHub repo.
- Set Start Command:
