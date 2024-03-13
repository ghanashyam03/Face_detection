# app.py

from flask import Flask, render_template, Response
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN, extract_face
import os

app = Flask(__name__)

# Load FaceNet model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(keep_all=True, device=device)

# Function to extract face embeddings
def extract_face_embeddings(image):
    faces = mtcnn(image)
    if faces is not None:
        embeddings = resnet(faces)
        return embeddings
    else:
        return None

# Function to calculate cosine similarity between embeddings
def cosine_similarity(embedding1, embedding2):
    embedding1_np = embedding1.detach().numpy().flatten()  # Flatten the embedding
    embedding2_np = embedding2.detach().numpy().flatten()  # Flatten the embedding
    dot_product = np.dot(embedding1_np, embedding2_np)
    norm1 = np.linalg.norm(embedding1_np)
    norm2 = np.linalg.norm(embedding2_np)
    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity

# Function to load and preprocess images from a directory
def load_images_and_extract_embeddings(directory):
    embeddings = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                embedding = extract_face_embeddings(image)
                if embedding is not None:
                    embeddings.append(embedding)
    return embeddings

# Load and preprocess embeddings of stored faces
stored_embeddings = load_images_and_extract_embeddings('C:/Users/DELL/Desktop/ghana/ai_detected_images/stored-faces')

# Function to check if there's a matched image and return it
def check_matched_image(frame_embedding, stored_embeddings, threshold=0.6):
    for stored_embedding in stored_embeddings:
        similarity_percentage = cosine_similarity(frame_embedding, stored_embedding)
        print("Similarity:", similarity_percentage)  # Debug statement
        if similarity_percentage > threshold:
            return True  # Match found
    
    return False  # No match found

# Function to process frames and check for matches
def process_frame(frame):
    # Extract face embeddings for the current frame
    frame_embedding = extract_face_embeddings(frame)
    if frame_embedding is not None:
        match_found = check_matched_image(frame_embedding, stored_embeddings)
        return match_found
    return False

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Generator function to get frames from the webcam
def webcam_gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for video feed
def gen():
    while True:
        frame = webcam_gen()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + next(frame) + b'\r\n')

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for face detection
@app.route('/detect')
def detect():
    match_found = False
    while not match_found:
        ret, frame = cap.read()
        if ret:
            match_found = process_frame(frame)
    return "Match found!"

if __name__ == '__main__':
    app.run(debug=True)
