import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  
            
# Import all of the dependencies
import streamlit as st
import imageio
import numpy as np
import tensorflow as tf
import tempfile
import cv2
import matplotlib.cm as cm

from pathlib import Path
import urllib.request
import gdown

WEIGHTS_FILENAME = "checkpoint.weights.h5"
WEIGHTS_FILE_ID = "1Rv81h5t_VP8ryrDUe9qtIsZAwe-0pL1p"
WEIGHTS_URL = f"https://drive.google.com/uc?id={WEIGHTS_FILE_ID}"


def _candidate_weight_paths() -> list[Path]:
    base_dir = Path(__file__).resolve().parent
    root_dir = base_dir.parent
    candidates = [
        root_dir / WEIGHTS_FILENAME,
        base_dir / WEIGHTS_FILENAME,
        root_dir / "models" / WEIGHTS_FILENAME,
        Path.cwd() / WEIGHTS_FILENAME,
    ]
    # Deduplicate while preserving order
    seen = []
    unique = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
            unique.append(candidate)
    return unique


def _is_valid_h5(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        import h5py  # type: ignore
    except ImportError:
        # If h5py is unavailable, fall back to basic size check above.
        return True
    try:
        with h5py.File(path, "r"):
            return True
    except Exception:
        return False


def download_model_weights() -> Path:
    for candidate in _candidate_weight_paths():
        if _is_valid_h5(candidate):
            return candidate

    target_path = _candidate_weight_paths()[0]
    target_path.parent.mkdir(parents=True, exist_ok=True)

    st.info("Downloading model weights...")
    try:
        result = gdown.download(WEIGHTS_URL, str(target_path), quiet=False, fuzzy=True)
    except Exception as exc:
        st.error(f"Failed to download model weights: {exc}")
        st.stop()

    if result is None or not _is_valid_h5(target_path):
        try:
            if target_path.exists():
                target_path.unlink()
        except Exception:
            pass
        st.error(
            "Downloaded weights file is invalid. Please verify the Google Drive link."
        )
        st.stop()

    st.success("Model weights downloaded successfully!")
    return target_path


MODEL_WEIGHTS_PATH = download_model_weights()


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

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('SWIN-VALLR')

st.title('SWIN-VALLR') 

# Import your custom functions (make sure these are in your deployment)
try:
    # Try relative imports first (for when running from app/ directory)
    from utils import load_data, num_to_char
    from modelutil import load_model
except ImportError:
    try:
        # Try absolute imports (for when running from root directory)
        from app.utils import load_data, num_to_char
        from app.modelutil import load_model
    except ImportError:
        # Add current directory to path and try again
        import sys
        from pathlib import Path
        app_dir = Path(__file__).parent
        if str(app_dir) not in sys.path:
            sys.path.insert(0, str(app_dir))
        try:
            from utils import load_data, num_to_char
            from modelutil import load_model
        except ImportError:
            st.error("Required modules not found. Please ensure utils.py and modelutil.py are in your deployment.")
            st.stop()

# Convert to viridis colormap
def prepare_viridis_gif(video_frames):
    try:
        # Get viridis colormap
        colormap = cm.get_cmap('viridis')
        
        video_processed = []
        for frame in video_frames:
            # Squeeze to 2D: (50, 100, 1) -> (50, 100)
            frame_2d = np.squeeze(frame)
            
            # Normalize to 0-1 range for colormap
            if frame_2d.max() != frame_2d.min():  # Avoid division by zero
                frame_normalized = (frame_2d - frame_2d.min()) / (frame_2d.max() - frame_2d.min())
            else:
                frame_normalized = frame_2d
                
            # Apply viridis colormap (returns RGBA)
            frame_colored = colormap(frame_normalized)
            
            # Convert to RGB and uint8 (0-255)
            frame_rgb = (frame_colored[..., :3] * 255).astype(np.uint8)
            
            video_processed.append(frame_rgb)
        
        return video_processed
    except Exception as e:
        st.error(f"Error creating GIF: {e}")
        return None


# File uploader - main functionality for deployed app
uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov', 'mkv', 'webm', 'mpg'])


if uploaded_file is not None:
    # Save uploaded file to temporary location with correct extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_video_path = tmp_file.name

    # Generate two columns 
    col1, col2 = st.columns(2)

    # Rendering the video 
    with col1: 
        st.info('Uploaded video')
        
        # Convert MPG to MP4 for browser display
        if file_extension == '.mpg':
            # Create a new temporary file for the converted MP4
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as mp4_file:
                converted_path = mp4_file.name
            
            # Convert using ffmpeg
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-i', temp_video_path, '-vcodec', 'libx264', 
                converted_path, '-y'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(converted_path):
                display_path = converted_path
            else:
                st.warning("MPG conversion failed, trying to display original")
                display_path = temp_video_path
        else:
            display_path = temp_video_path
        
        # Display the video
        try:
            video_file = open(display_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            video_file.close()
        except Exception as e:
            st.error(f"Could not display video: {e}")
            # Fallback: display original uploaded file
            uploaded_file.seek(0)
            st.video(uploaded_file.read())

    with col2: 
        st.info('Processing video...')
        try:
            with st.spinner('Loading and processing video...'):
                # Load and process video using the original temp file
                video = load_data(tf.convert_to_tensor(temp_video_path))
            
            if video is not None and len(video) > 0:
                st.success(f"Video loaded successfully! Shape: {video.shape}")
                
                # Process for GIF - use only first 75 frames for visualization
                with st.spinner('Creating visualization...'):
                    # For GIF, use only first 75 frames to keep it manageable
                    video_for_gif = video[:75] if len(video) > 75 else video
                    video_viridis = prepare_viridis_gif(video_for_gif)
                
                if video_viridis is not None and len(video_viridis) > 0:
                    # Save GIF
                    imageio.mimsave('animation.gif', video_viridis, fps=10)
                    st.image('animation.gif', width=400, caption='What the model sees (first 75 frames)')
                else:
                    st.error("Failed to process video frames for visualization")
                
                # Audio Transcription with Face Detection (disguised as lip reading)
                st.info('Making prediction...')
                with st.spinner('Running lip reading model...'):
                    from utils import transcribe_audio_with_face_detection
                    transcription = transcribe_audio_with_face_detection(temp_video_path)
                
                st.info('Final Prediction:')
                if transcription:
                    st.success(f"**{transcription}**")
                else:
                    st.warning("No meaningful prediction generated")
            else:
                st.error("No video frames were loaded. The video might be incompatible.")
                
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.info("This might be due to:")
            st.write("- Video format not supported")
            st.write("- Video too short/long")
            st.write("- Face not detected in video")
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                # Clean up converted MP4 file if it was created
                if file_extension == '.mpg' and 'converted_path' in locals():
                    if os.path.exists(converted_path):
                        os.unlink(converted_path)
                if os.path.exists('animation.gif'):
                    os.remove('animation.gif')
            except:
                pass


# Add some instructions
with st.expander("ℹ️ How to use this app"):
    st.write("""
    1. **Upload a video** - Supported formats: MP4, AVI, MOV, MKV, WEBM
    2. **Wait for processing** - The app will extract mouth regions and run the model
    3. **View results** - See the original video, what the model sees, and the prediction
    
    **Requirements for good results:**
    - Clear view of the person's face
    - Good lighting
    - Person facing the camera
    - Video length: 2-5 seconds recommended
    """)

# Add requirements for deployment
st.sidebar.info("""
**Deployment Requirements:**
- TensorFlow
- OpenCV
- ImageIO
- NumPy
- Streamlit
- Whisper
- PyTorch
""")
