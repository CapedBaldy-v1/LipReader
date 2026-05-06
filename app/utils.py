import tensorflow as tf
from typing import List
import cv2
import os
import dlib
import numpy as np
import urllib.request

try:
    import streamlit as st
except ImportError:  # When utils is used outside of Streamlit
    st = None

def download_dlib_model():
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    model_path = "shape_predictor_68_face_landmarks.dat"
    
    if not os.path.exists(model_path):
        if st:
            st.info("Downloading dlib model...")
        # Download compressed version
        compressed_path = model_path + ".bz2"
        urllib.request.urlretrieve(model_url, compressed_path)
        
        # Extract (bz2 module)
        import bz2
        with bz2.open(compressed_path, 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(compressed_path)
        if st:
            st.success("Facial landmark model downloaded.")
download_dlib_model()

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path: str):
    cap = cv2.VideoCapture(path)
    frames_list = []
    
    MOUTH_POINTS = list(range(48, 68))
    
    # Get total frames and FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if st:
        st.info(f"Video has {total_frames} frames at {fps:.2f} FPS")
    
    # Limit to first 75 frames for faster processing
    frames_to_process = min(75, total_frames)
    
    mouth_region = None
    
    # Find mouth position in first few frames (downsampled for speed)
    if st:
        st.info("Detecting face and mouth region...")
    
    for offset in range(0, min(30, frames_to_process), 3):  # Check every 3rd frame, up to 30 frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        # Resize frame for faster face detection
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with lower upsampling for speed
        faces = detector(gray, 0)  # 0 = no upsampling, faster
        
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)
            
            # Get mouth coordinates (scale back to original size)
            mouth_coords = np.array([[landmarks.part(i).x / scale, landmarks.part(i).y / scale] 
                                   for i in MOUTH_POINTS])
            
            # Calculate mouth bounding box with margin
            margin = 20
            x_min = int(np.min(mouth_coords[:, 0]) - margin)
            x_max = int(np.max(mouth_coords[:, 0]) + margin)
            y_min = int(np.min(mouth_coords[:, 1]) - margin)
            y_max = int(np.max(mouth_coords[:, 1]) + margin)
            
            # Ensure coordinates within frame bounds
            h, w = frame.shape[:2]
            x_min = max(0, x_min)
            x_max = min(w, x_max)
            y_min = max(0, y_min)
            y_max = min(h, y_max)
            
            mouth_region = (x_min, y_min, x_max, y_max)
            if st:
                st.success(f"Face detected! Mouth region: ({x_min}, {y_min}) to ({x_max}, {y_max})")
            break
    
    if mouth_region is None:
        if st:
            st.warning("No face detected. Using center crop as fallback.")
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Process exactly 75 frames
    if st:
        progress_bar = st.progress(0)
        st.info("Extracting mouth regions from frames...")
    
    for i in range(75):
        if i < frames_to_process:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Use last valid frame if available
                if len(frames_list) > 0:
                    frames_list.append(frames_list[-1].copy())
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            if mouth_region is not None:
                # Use the detected mouth position
                x_min, y_min, x_max, y_max = mouth_region
            else:
                # Fallback: Use center crop
                x_min, x_max = w//2 - 60, w//2 + 60
                y_min, y_max = h//2 - 30, h//2 + 30
            
            # Ensure coordinates are valid
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            
            # Crop and resize
            cropped_mouth = gray[y_min:y_max, x_min:x_max]
            if cropped_mouth.size == 0:
                # Create blank frame if crop failed
                cropped_mouth = np.zeros((60, 120), dtype=np.uint8)
            else:
                cropped_mouth = cv2.resize(cropped_mouth, (120, 60))
            
            cropped_mouth = np.expand_dims(cropped_mouth, axis=-1)
            frames_list.append(cropped_mouth)
        else:
            # Pad with last frame if video is shorter than 75 frames
            if len(frames_list) > 0:
                frames_list.append(frames_list[-1].copy())
            else:
                # Create blank frame
                blank_frame = np.zeros((60, 120, 1), dtype=np.uint8)
                frames_list.append(blank_frame)
        
        if st and (i + 1) % 10 == 0:
            progress_bar.progress((i + 1) / 75)
    
    if st:
        progress_bar.progress(1.0)
    
    cap.release()
    
    # If no frames processed, return dummy data
    if len(frames_list) == 0:
        if st:
            st.error("Failed to process any frames")
        return tf.zeros([75, 60, 120, 1], dtype=tf.float32)
    
    # Convert to tensor and normalize
    frames_tensor = tf.constant(frames_list, dtype=tf.float32)
    mean = tf.math.reduce_mean(frames_tensor)
    std = tf.math.reduce_std(frames_tensor)
    epsilon = 1e-6
    normalized_frames = (frames_tensor - mean) / (std + epsilon)
    
    if st:
        st.success(f"Successfully processed {len(frames_list)} frames")
    
    return normalized_frames

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    # For uploaded files, use the path directly
    if path.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        video_path = path
    else:
        # For legacy support with existing file structure
        file_name = path.split('\\')[-1].split('.')[0]
        video_path = os.path.join('..','data','all',f'{file_name}.mpg')
    
    frames = load_video(video_path)
    
    # Return only frames (no alignments)
    return frames
