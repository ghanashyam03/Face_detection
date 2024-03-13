import cv2
import numpy as np
import psycopg2
from PIL import Image as PILImage
from imgbeddings import imgbeddings
from concurrent.futures import ThreadPoolExecutor

# Initialize the video capture object
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
# Initialize imgbeddings object
ibed = imgbeddings()

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity

# Function to check if there's a matched image and return it
def check_matched_image(frame_embedding, threshold=0.90):
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect("postgres://avnadmin:AVNS_nHR6CTga0AQ2C57qKkK@pg-23f81deb-harirenjith123face.a.aivencloud.com:20445/defaultdb?sslmode=require")

        # Create a cursor object
        cur = conn.cursor()

        # Execute query to find images with embeddings similar to the current frame
        cur.execute("SELECT embedding FROM pictures;")
        rows = cur.fetchall()

        for row in rows:
            try:
                db_embedding = np.fromstring(row[0][1:-1], dtype=np.float32, sep=',')  # Convert embedding string to numpy array
                similarity_percentage = cosine_similarity(frame_embedding, db_embedding)
                print("Similarity:", similarity_percentage)  # Debug statement
                if similarity_percentage > threshold:
                    return True  # Match found
            except ValueError as e:
                print("Error converting embedding to float:", e)
                continue

        cur.close()
    except psycopg2.Error as e:
        print("Unable to connect to the database:", e)
    finally:
        if conn:
            conn.close()

    return False  # No match found

# Function to process frames and check for matches
def process_frame(frame):
    # Calculate embeddings for the current frame
    frame_embedding = ibed.to_embeddings(PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    match_found = check_matched_image(frame_embedding)
    if match_found:
        return True
    else:
        return False

# Function to continuously process frames from the webcam
def process_frames():
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                ret, frame = cap.read()
                if ret:
                    future = executor.submit(process_frame, frame)
                    match_found = future.result()
                    
                    if match_found:
                        text = "MATCH"
                        color = (0, 255, 0)  # Green color for match
                    else:
                        text = "NO MATCH"
                        color = (0, 0, 255)  # Red color for no match
                    
                    # Add text to the frame
                    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    cv2.imshow("video", frame)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()
        cap.release()

# Start processing frames
process_frames()
