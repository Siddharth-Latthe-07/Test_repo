from fastapi import FastAPI, UploadFile, File, HTTPException
import face_recognition
import numpy as np
import faiss
import os

app = FastAPI()

# Paths for FAISS index and metadata
FAISS_INDEX_FILE = "encodings.index"
IMAGE_IDS_FILE = "image_ids.npy"

# Initialize FAISS index
D = 128  # Dimensionality of face encodings
if os.path.exists(FAISS_INDEX_FILE):
    index = faiss.read_index(FAISS_INDEX_FILE)
else:
    index = faiss.IndexFlatL2(D)  # L2 distance-based FAISS index

# Load image IDs if they exist
if os.path.exists(IMAGE_IDS_FILE):
    image_ids = np.load(IMAGE_IDS_FILE).tolist()
else:
    image_ids = []

# Endpoint to generate and store encodings
@app.post("/generate-encoding")
async def generate_encoding(file: UploadFile = File(...), image_id: str = ""):
    if not image_id:
        raise HTTPException(status_code=400, detail="Image ID is required.")

    # Read image
    image_data = await file.read()
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

    # Add encoding to FAISS index and update image IDs
    index.add(np.array([face_encoding]))
    image_ids.append(image_id)

    # Save updated FAISS index and image IDs
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(IMAGE_IDS_FILE, np.array(image_ids))

    return {"message": "Encoding added successfully!", "image_id": image_id}

# Endpoint to search for matches
@app.post("/search")
async def search(file: UploadFile = File(...)):
    if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(IMAGE_IDS_FILE):
        raise HTTPException(status_code=400, detail="Encodings or image IDs not initialized.")

    # Read image
    image_data = await file.read()
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
    distances, indices = index.search(np.array([face_encoding]), k=5)  # Top 5 matches
    matches = []
    for i, distance in enumerate(distances[0]):
        if distance < 0.6:  # Set a threshold for matching
            matches.append({
                "image_id": image_ids[indices[0][i]],
                "distance": float(distance)
            })

    if matches:
        return {"message": "Matches found!", "matches": matches}
    else:
        return {"message": "No matches found."}
      
