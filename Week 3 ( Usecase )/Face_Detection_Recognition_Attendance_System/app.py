"""Made with fun by Prathamesh..."""
import io
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Env loading
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass

import numpy as np
if hasattr(np, "__version__") and int(np.__version__.split(".")[0]) >= 2:
    print("ERROR: NumPy 2.x is not compatible. Run: pip install 'numpy>=1.24,<2'", file=sys.stderr)
    sys.exit(1)

import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer

import database as db
import face_logic as fl

# Default recognition threshold
DEFAULT_THRESHOLD = 0.38

# --- Caching Resources ---

@st.cache_resource
def get_db_connection():
    """Initializes and caches the MongoDB connection."""
    return db.get_db()

@st.cache_resource
def load_detector():
    return fl.get_detector()

@st.cache_resource
def load_arcface():
    return fl.get_arcface_model()

def load_embeddings_cache(db_conn):
    """Load enrollments into session state, passing the cached db connection."""
    if "embeddings_cache" in st.session_state and st.session_state.embeddings_cache is not None:
        return st.session_state.embeddings_cache
    try:
        emb_list, label_list = db.get_all_enrollments_for_cache(db_conn)
        if not emb_list:
            st.session_state.embeddings_cache = (np.zeros((0, 512), dtype=np.float32), [])
        else:
            db_emb = np.array(emb_list, dtype=np.float32)
            norms = np.linalg.norm(db_emb, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            db_emb = db_emb / norms
            st.session_state.embeddings_cache = (db_emb, label_list)
        return st.session_state.embeddings_cache
    except Exception as e:
        st.error(f"Could not load enrollments: {e}")
        return (np.zeros((0, 512), dtype=np.float32), [])

# --- Pages ---

def enrollment_page():
    st.subheader("Multi-Pose Enrollment")
    
    # Initialize session state for multi-step enrollment
    if "enroll_step" not in st.session_state:
        st.session_state.enroll_step = 0
        st.session_state.temp_embeddings = []

    try:
        db_conn = get_db_connection()
    except Exception as e:
        st.error(f"Database Error: {e}")
        return

    user_id = st.text_input("User ID ", placeholder="e.g. UCS22M1019")
    user_name = st.text_input("Name ", placeholder="e.g. Prathamesh")
    
    # Define the 3 poses
    poses = ["Straight", "Left Profile", "Right Profile"]
    current_step = st.session_state.enroll_step

    if current_step < 3:
        st.info(f"Step {current_step + 1}: Please look **{poses[current_step]}**")
        photo = st.camera_input(f"Capture {poses[current_step]} face", key=f"cam_{current_step}")
        
        if photo:
            try:
                detector = load_detector()
                arcface = load_arcface()
                
                # Image processing
                bytes_data = photo.getvalue()
                nparr = np.frombuffer(bytes_data, np.uint8)
                image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                face_rgb = fl.extract_largest_face(image_bgr, detector)
                if face_rgb is None:
                    st.error("Face not detected. Try again.")
                else:
                    embedding = fl.get_embedding(face_rgb, arcface)
                    if embedding is not None:
                        st.session_state.temp_embeddings.append(embedding.tolist())
                        st.session_state.enroll_step += 1
                        st.rerun()  # Move to next step
            except Exception as e:
                st.error(f"Error: {e}")

    # Final Enrollment Step
    if st.session_state.enroll_step == 3:
        st.success("All 3 poses captured!")
        if st.button("Complete Enrollment") and user_id:
            db.insert_enrollment(
                user_id.strip(), 
                user_name.strip(), 
                st.session_state.temp_embeddings, 
                db=db_conn
            )
            st.success(f"Successfully enrolled **{user_name}** with 3 poses.")
            
            # Reset state for next user
            st.session_state.enroll_step = 0
            st.session_state.temp_embeddings = []
            if "embeddings_cache" in st.session_state:
                del st.session_state.embeddings_cache



def live_attendance_page():
    st.subheader("Live Attendance (Photo Capture)")
    threshold = 0.38

    try:
        db_conn = get_db_connection()
        detector = load_detector()
        arcface = load_arcface()
    except Exception as e:
        st.error(f"System Error: {e}")
        return

    db_emb, db_labels = load_embeddings_cache(db_conn)
    if not db_labels:
        st.warning("No users enrolled yet.")
        return

    enroll_coll = db_conn["enrollments"]
    all_docs = enroll_coll.find({}, {"user_id": 1, "name": 1})
    id_to_name = {doc["user_id"]: doc.get("name", doc["user_id"]) for doc in all_docs}
    photo = st.camera_input("Position your face and take a photo")

    if photo:
        try:
            image_bgr = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            with st.spinner("Recognizing..."):
                output_img, recognized_list = fl.process_frame(
                    image_bgr, detector, arcface, db_emb, db_labels, name_map=id_to_name, threshold=threshold
                )
            
            st.image(output_img, channels="BGR", caption="Result")
            
            if not recognized_list:
                st.error("No faces detected.")
            else:
                for user_id, score in recognized_list:
                    if user_id != "Unknown":
                        # Mark Attendance
                        success = db.insert_attendance(user_id, db=db_conn)
                        
                        # Fetch the name for display
                        user_info = db.get_enrollment_by_user_id(user_id, db=db_conn)
                        display_name = user_info.get("name", user_id) if user_info else user_id
                        
                        if success:
                            st.success(f"✅ Attendance Marked: **{display_name}** ({user_id})")
                        else:
                            st.info(f"ℹ️ **{display_name}** already marked recently.")
                    else:
                        st.error("Face not recognized.")
        except Exception as e:
            st.error(f"Error: {e}")

import pytz

def reports_page():
    st.subheader("Attendance Reports")
    ist = pytz.timezone('Asia/Kolkata')
    today = datetime.now(ist).date()
    
    try:
        db_conn = get_db_connection()
    except Exception as e:
        st.error(f"DB Error: {e}")
        return
        
    col1, col2 = st.columns(2)
    with col1:
        from_date = st.date_input("From", value=today)
    with col2:
        to_date = st.date_input("To", value=today)
        
    if from_date > to_date:
        st.error("Invalid date range.")
        return
        
    # Convert dates to localized datetimes
    from_dt = datetime.combine(from_date, datetime.min.time())
    to_dt = datetime.combine(to_date, datetime.max.time())
    
    records = db.get_attendance(from_dt, to_dt, db=db_conn)
    
    if records:
        rows = [{
            "Date": r["timestamp"].strftime("%Y-%m-%d"),
            "Time": r["timestamp"].strftime("%I:%M %p"), # 12-hour format
            "User ID": r["user_id"],
            "Name": r.get("name", r["user_id"])
        } for r in records]
        
        st.dataframe(rows, use_container_width=True)
        
        # CSV Export
        buf = io.StringIO()
        import csv
        writer = csv.DictWriter(buf, fieldnames=["Date", "Time", "User ID", "Name"])
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("Download CSV", buf.getvalue(), f"attendance_{today}.csv", "text/csv")
    else:
        st.info("No records found for the selected dates.")

def main():
    st.set_page_config(page_title="Face Attendance", page_icon="📋", layout="wide")
    st.title("Face Detection & Recognition Attendance System")
    
    page = st.sidebar.radio("Navigation", ["Enrollment", "Live Attendance", "Reports"])
    
    if page == "Enrollment":
        enrollment_page()
    elif page == "Live Attendance":
        live_attendance_page()
    elif page == "Reports":
        reports_page()

if __name__ == "__main__":
    main()