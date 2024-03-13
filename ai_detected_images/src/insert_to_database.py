# importing the required libraries
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os

# connecting to the database - replace the SERVICE URI with the service URI
conn = psycopg2.connect("postgres://avnadmin:AVNS_nHR6CTga0AQ2C57qKkK@pg-23f81deb-harirenjith123face.a.aivencloud.com:20445/defaultdb?sslmode=require")
stored_faces_dir = "D:/extra/ai_detected_images/src/stored-faces"

for filename in os.listdir("D:/extra/ai_detected_images/stored-faces"):
    # opening the image
    img = Image.open("D:/extra/ai_detected_images/stored-faces/" + filename)
    # loading the `imgbeddings`
    ibed = imgbeddings()
    # calculating the embeddings
    embedding = ibed.to_embeddings(img)
    cur = conn.cursor()
    cur.execute("INSERT INTO pictures values (%s,%s)", (filename, embedding[0].tolist()))
    print(filename)
conn.commit()
