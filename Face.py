from fastapi import FastAPI, UploadFile, File, HTTPException
import face_recognition
import numpy as np
import faiss
import os

app = FastAPI()

# Path for FAISS index file
FAISS_INDEX_FILE = "encodings_with_ids.index"

# Dimensionality of face encodings
D = 128

# Initialize FAISS index
if os.path.exists(FAISS_INDEX_FILE):
    print("Loading existing FAISS index...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    id_map = faiss.IndexIDMap2(index)  # Wrap with IndexIDMap2
else:
    print("Creating new FAISS index...")
    flat_index = faiss.IndexFlatL2(D)
    id_map = faiss.IndexIDMap(flat_index)

# Utility function to generate a face encoding from an image
def generate_face_encoding(image_path: str):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        raise ValueError(f"No face detected in image: {image_path}")

    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
    return face_encoding

# Endpoint to generate encodings for a folder
@app.post("/generate-encodings-folder", status_code=201)
def generate_encodings_folder(folder_path: str):
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail="Folder path does not exist.")

    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        raise HTTPException(status_code=400, detail="No valid image files found in the folder.")

    added_images = 0
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image_id = hash(image_path) % (10**8)  # Generate unique ID for the image

        # Skip already encoded images
        if id_map.is_trained and image_id in id_map.id_map:
            continue

        try:
            encoding = generate_face_encoding(image_path)
            id_map.add_with_ids(np.array([encoding], dtype="float32"), np.array([image_id], dtype="int64"))
            added_images += 1
        except ValueError as e:
            print(f"Skipping {image_path}: {e}")

    # Save updated FAISS index
    faiss.write_index(id_map.index, FAISS_INDEX_FILE)

    return {"message": f"Encodings generated successfully for {added_images} new images."}

# Endpoint to generate encoding for a single image
@app.post("/generate-encoding", status_code=201)
def generate_encoding(file: UploadFile = File(...), image_id: str = ""):
    if not image_id:
        raise HTTPException(status_code=400, detail="Image ID is required.")

    # Save uploaded file temporarily
    image_data = file.file.read()
    with open("temp_image.jpg", "wb") as temp_file:
        temp_file.write(image_data)

    try:
        encoding = generate_face_encoding("temp_image.jpg")
        id_map.add_with_ids(np.array([encoding], dtype="float32"), np.array([int(image_id)], dtype="int64"))
    except ValueError as e:
        os.remove("temp_image.jpg")
        raise HTTPException(status_code=400, detail=str(e))

    os.remove("temp_image.jpg")

    # Save updated FAISS index
    faiss.write_index(id_map.index, FAISS_INDEX_FILE)

    return {"message": "Encoding added successfully!", "image_id": image_id}

# Endpoint to search for matches
@app.post("/search", status_code=200)
def search(file: UploadFile = File(...)):
    if not os.path.exists(FAISS_INDEX_FILE):
        raise HTTPException(status_code=400, detail="FAISS index is not initialized.")

    # Save uploaded file temporarily
    image_data = file.file.read()
    with open("temp_image.jpg", "wb") as temp_file:
        temp_file.write(image_data)

    try:
        encoding = generate_face_encoding("temp_image.jpg")
    except ValueError as e:
        os.remove("temp_image.jpg")
        raise HTTPException(status_code=400, detail=str(e))

    os.remove("temp_image.jpg")

    # Search the FAISS index
    distances, indices = id_map.search(np.array([encoding], dtype="float32"), k=5)
    matches = []
    for i, distance in enumerate(distances[0]):
        if distance < 0.6:  # Threshold for matching
            matches.append({"image_id": str(indices[i]), "distance": distance})

    if matches:
        return {"message": "Matches found!", "matches": matches}
    else:
        raise HTTPException(status_code=404, detail="No matches found.")
        
