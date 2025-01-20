from fastapi import FastAPI, UploadFile, File, HTTPException
import face_recognition
import numpy as np
import faiss
import os

app = FastAPI()

# Path for FAISS index file
FAISS_INDEX_FILE = "encodings_with_ids.index"

# Initialize FAISS index with metadata support
D = 128  # Dimensionality of face encodings
index = None

if os.path.exists(FAISS_INDEX_FILE):
    index = faiss.read_index(FAISS_INDEX_FILE)
    id_map = faiss.IndexIDMap(index)  # IDMap for storing image IDs
else:
    flat_index = faiss.IndexFlatL2(D)  # L2 distance-based FAISS index
    id_map = faiss.IndexIDMap(flat_index)

# Endpoint to generate and store encodings
@app.post("/generate-encoding", status_code=201)
def generate_encoding(file: UploadFile = File(...), image_id: str = ""):
    if not image_id:
        raise HTTPException(status_code=400, detail="Image ID is required.")

    # Read image
    image_data = file.file.read()
    with open("temp_image.jpg", "wb") as temp_file:
        temp_file.write(image_data)

    # Generate face encoding
    image = face_recognition.load_image_file("temp_image.jpg")
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        os.remove("temp_image.jpg")
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
    os.remove("temp_image.jpg")

    # Add encoding and image ID to the FAISS index
    id_map.add_with_ids(np.array(face_encoding, dtype="float32"), np.array([int(image_id)], dtype="int64"))

    # Save updated FAISS index
    faiss.write_index(id_map.index, FAISS_INDEX_FILE)

    return {"message": "Encoding added successfully!", "image_id": image_id}

# Endpoint to search for matches
@app.post("/search", status_code=200)
def search(file: UploadFile = File(...)):
    if not os.path.exists(FAISS_INDEX_FILE):
        raise HTTPException(status_code=400, detail="FAISS index is not initialized.")

    # Read image
    image_data = file.file.read()
    with open("temp_image.jpg", "wb") as temp_file:
        temp_file.write(image_data)

    # Generate face encoding
    image = face_recognition.load_image_file("temp_image.jpg")
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        os.remove("temp_image.jpg")
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image.")

    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
    os.remove("temp_image.jpg")

    # Search FAISS index
    distances, indices = id_map.search(np.array(face_encoding, dtype="float32"), k=5)  # Top 5 matches
    matches = []
    for i, distance in enumerate(distances[0]):
        if distance < 0.6:  # Set a threshold for matching
            matches.append({
                "image_id": str(indices[0][i])
                #"distance": distance,
            })

    if matches:
        return {"message": "Matches found!", "matches": matches}
    else:
        raise HTTPException(status_code=404, detail="No matches found.")



curl -X POST "http://127.0.0.1:8000/generate-encoding" \
-F "file=@path/to/image.jpg" \
-F "image_id=123"


curl -X POST "http://127.0.0.1:8000/generate-encoding?image_id=123" \
-F "file=@path/to/image.jpg"


curl -X POST "http://127.0.0.1:8000/search" \
-F "file=@path/to/query_image.jpg"


