import numpy as np
import mediapipe as mp
import cv2
from tqdm import tqdm
from pathlib import Path
import landmark_pb2
import os
from moviepy.editor import AudioFileClip, VideoFileClip
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    refine_landmarks=True, 
    max_num_faces=1
)

def getFaceLandmarksFromFrame(frame, isBGR = False):
    if isBGR : 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame)
    landmarks = []
    if result.multi_face_landmarks and len(result.multi_face_landmarks) > 0:
        landmarks = result.multi_face_landmarks[0]
        landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark]
    return landmarks

def center_on_centroid(X):
    """
    Center landmarks on the centroid (mean) of the face without removing rotation.
    X: Input landmarks of shape (nframes, 478, 3)
    
    Returns:
        Landmarks centered on the frame's origin (0,0,0) for each frame, without altering rotation.
    """
    nframes = X.shape[0]
    centered_landmarks = np.zeros_like(X)
    
    for i in range(nframes):
        # Compute the centroid (mean of all landmarks) for each frame
        centroid = np.mean(X[i], axis=0)
        
        # Center the landmarks by subtracting the centroid
        centered_landmarks[i] = X[i] - centroid
    
    return centered_landmarks

def normalize_landmarks(X):
    """
    Normalize landmarks to a [0, 1] range after centering on the centroid of the face,
    while preserving rotation.
    X: Input landmarks of shape (nframes, 478, 3)
    
    Returns:
        Normalized landmarks with the same shape as X.
    """
    nframes = X.shape[0]
    normalized_landmarks = np.zeros_like(X)
    
    for i in range(nframes):
        # Get the bounding box for the centered landmarks
        min_coords = np.min(X[i], axis=0)
        max_coords = np.max(X[i], axis=0)
        bounding_box_size = max_coords - min_coords
        
        # Avoid division by zero
        bounding_box_size[bounding_box_size == 0] = 1
        
        # Normalize landmarks to fit in the [0, 1] range without altering rotation
        normalized_landmarks[i] = (X[i] - min_coords) / bounding_box_size
    
    return normalized_landmarks

def get_specifications(alpha, thickness, r):
    return {
        "left_eyebrow" : {
            "spec" : mp_drawing.DrawingSpec(color=(0, alpha, 0), thickness=thickness, circle_radius=r),
            "connections" : mp_face_mesh.FACEMESH_LEFT_EYEBROW
        },
        "right_eyebrow" : {
            "spec" : mp_drawing.DrawingSpec(color=(0, alpha, 0), thickness=thickness, circle_radius=r),
            "connections" : mp_face_mesh.FACEMESH_RIGHT_EYEBROW
        },
        "face_outline" : {
            "spec" : mp_drawing.DrawingSpec(color=(alpha, 0, 0), thickness=thickness, circle_radius=r),
            "connections" : mp_face_mesh.FACEMESH_FACE_OVAL
        },
        "nose" : {
            "spec" : mp_drawing.DrawingSpec(color=(0, 0, alpha), thickness=thickness, circle_radius=r),
            "connections" : mp_face_mesh.FACEMESH_NOSE
        },
        "left_eye" : {
            "spec" : mp_drawing.DrawingSpec(color=(0, alpha, alpha), thickness=thickness, circle_radius=r),
            "connections" : mp_face_mesh.FACEMESH_LEFT_EYE
        },
        "right_eye" : {
            "spec" : mp_drawing.DrawingSpec(color=(0, alpha, alpha), thickness=thickness, circle_radius=r),
            "connections" : mp_face_mesh.FACEMESH_RIGHT_EYE
        },
        "irises" : {
            "spec" : mp_drawing.DrawingSpec(color=(0, alpha, alpha), thickness=thickness, circle_radius=r),
            "connections" : mp_face_mesh.FACEMESH_IRISES
        },
        "lips" :  {
            "spec" : mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=thickness, circle_radius=r),
            "connections" : mp_face_mesh.FACEMESH_LIPS
        },
    }

def draw_landmarks(frames, output_dir, w, h, alpha, thickness, r):
    output_dir.mkdir(parents=True, exist_ok=True)
    Specifications = get_specifications(alpha, thickness, r)
    for j, frame in tqdm(enumerate(frames)):
        img = np.full((w, h, 3), (255, 255, 255), np.uint8)
        landmark_lst = [{'x': lm[0], 'y': lm[1], 'z': lm[2]} for lm in frame]
        landmark_subset = landmark_pb2.NormalizedLandmarkList(landmark=landmark_lst)
        for val in Specifications.values():
            connections, spec = val["connections"], val["spec"]
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=landmark_subset,
                connections=connections,
                landmark_drawing_spec=None,
                connection_drawing_spec=spec
            )
        cv2.imwrite(str(output_dir / f"{j}.png"), img)
        
def process_video(video_path, output_path, alpha, thickness, r):
    video_clip = VideoFileClip(video_path)
    w, h = video_clip.size
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        found, frame = cap.read()
        if not found:
            break
        frames.append(frame)
    cap.release()
        
    landmark_frames = []
    for frame in tqdm(frames):
        landmarks = getFaceLandmarksFromFrame(frame, isBGR=True)
        
        # Added centering and normalization
        if landmarks:
            landmarks = np.array(landmarks).reshape(1, -1, 3)  # Reshape to (1, 478, 3)
            landmarks = center_on_centroid(landmarks)
            landmarks = normalize_landmarks(landmarks)
            landmarks = landmarks[0]  # Reshape back to (478, 3)
        
        landmark_frames.append(landmarks)
    
    temp_dir = Path("./temp_frames")
    temp_dir.mkdir(parents=True, exist_ok=True)
    draw_landmarks(landmark_frames, temp_dir, w, h, alpha, thickness, r)
    print(f"Landmarks drawn")
    
    audio_path = os.path.join(temp_dir, "audio.wav")
    video_clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
    
    fps = len(frames) / video_clip.duration
    print(f"Using fps = {fps}")
    def create_video_from_frames(path_in, path_out):
        frame_files = sorted([img for img in os.listdir(path_in) if img.endswith(".png")], key=lambda x: int(x.split('.')[0]))
        frame_array = [cv2.imread(os.path.join(path_in, file)) for file in frame_files]
        height, width, _ = frame_array[0].shape
        video_out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        for frame in frame_array:
            video_out.write(frame)
        video_out.release()

    def overlay_audio(video_path, audio_path, output_path):
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        final_video = video.set_audio(audio)
        final_video.write_videofile(output_path)
    
    temp_output = "./temp_frames/video.mp4"
    create_video_from_frames(temp_dir, temp_output)
    overlay_audio(temp_output, audio_path, output_path)
        
    for filetype in ["png", "wav", "mp4"]:
        for file in temp_dir.glob(f"*.{filetype}"):
            file.unlink()


def process_image(image_path, output_path, alpha, thickness, r):
    frame = cv2.imread(image_path)
    h, w, _ = frame.shape
    landmarks = getFaceLandmarksFromFrame(frame, isBGR=True)
    #Added centering and Normalizations: 
    # Step 1: Center the landmarks on the centroid (center of the face)
    landmarks = center_on_centroid(landmarks)

    # Step 2: Normalize landmarks to [0, 1] range without altering rotation
    landmarks = normalize_landmarks(landmarks)
    #END OF CHANGE
    
    img = np.full((h, w, 3), (255, 255, 255), np.uint8)
    landmark_lst = [{'x': lm[0], 'y': lm[1], 'z': lm[2]} for lm in landmarks]
    
    landmark_subset = landmark_pb2.NormalizedLandmarkList(landmark=landmark_lst)
    
    Specifications = get_specifications(alpha, thickness, r)
    for val in Specifications.values():
        connections, spec = val["connections"], val["spec"]
        mp_drawing.draw_landmarks(
            image=img,
            landmark_list=landmark_subset,
            connections=connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=spec
        )
    
    cv2.imwrite(output_path, img)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images or videos to extract and draw face landmarks.")
    parser.add_argument("input_path", type=str, help="Path to the input image or video file.")
    parser.add_argument("output_path", type=str, help="Path to save the output image or video file.")
    parser.add_argument("--type", type=str, choices=["image", "video"], required=True, help="Specify whether the input is an image or a video.")
    parser.add_argument("--alpha", type=int, default=200, help="Alpha value for drawing specifications.")
    parser.add_argument("--thickness", type=int, default=3, help="Thickness value for drawing specifications.")
    parser.add_argument("--radius", type=int, default=2, help="Circle radius value for drawing specifications.")
    
    args = parser.parse_args()
    
    if args.type == "image":
        process_image(args.input_path, args.output_path, args.alpha, args.thickness, args.radius)
    elif args.type == "video":
        process_video(args.input_path, args.output_path, args.alpha, args.thickness, args.radius)