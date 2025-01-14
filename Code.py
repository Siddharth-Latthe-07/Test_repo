from fastapi import FastAPI, File, UploadFile, HTTPException
import face_recognition
import os
import faiss
import numpy as np

app = FastAPI()

DATASET_PATH = "/home/siddharth/Desktop/face_recg/dataset"  # Update this to your dataset path
INDEX_FILE = "faiss_index.bin"
IMAGES_FILE = "image_names.npy"

# Utility function to create or update the FAISS index
def update_faiss_index(dataset_path, index_file, images_file):
    # Initialize or load FAISS index
    if os.path.exists(index_file) and os.path.exists(images_file):
        index = faiss.read_index(index_file)
        existing_images = np.load(images_file).tolist()
    else:
        index = faiss.IndexFlatL2(128)  # 128-dimensional encodings
        existing_images = []

    # Process new images
    new_encodings = []
    new_images = []
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and filename not in existing_images:
            image_path = os.path.join(dataset_path, filename)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                new_encodings.append(face_encoding)
                new_images.append(filename)
            else:
                print(f"No face found in {filename}")

    # Add new encodings to the FAISS index
    if new_encodings:
        index.add(np.array(new_encodings, dtype=np.float32))

    # Update image names
    updated_images = existing_images + new_images

    # Save updated FAISS index and image names
    faiss.write_index(index, index_file)
    np.save(images_file, np.array(updated_images))

    return len(new_images), len(existing_images)

# Endpoint to generate/update encodings
@app.post("/generate-encodings")
async def generate_or_update_encodings_endpoint():
    new_count, existing_count = update_faiss_index(DATASET_PATH, INDEX_FILE, IMAGES_FILE)
    return {
        "message": "Encodings updated successfully!",
        "new_images_processed": new_count,
        "total_images_in_dataset": new_count + existing_count
    }

# Endpoint to upload and match
@app.post("/upload-and-match")
async def upload_and_match(file: UploadFile = File(...)):
    if not os.path.exists(INDEX_FILE) or not os.path.exists(IMAGES_FILE):
        raise HTTPException(status_code=400, detail="FAISS index or image names file not found. Generate encodings first.")

    # Load FAISS index and image names
    index = faiss.read_index(INDEX_FILE)
    image_names = np.load(IMAGES_FILE).tolist()

    # Load user image
    image_data = await file.read()
    with open("temp_uploaded_image.jpg", "wb") as temp_file:
        temp_file.write(image_data)
    user_image = face_recognition.load_image_file("temp_uploaded_image.jpg")

    user_face_locations = face_recognition.face_locations(user_image)
    if len(user_face_locations) == 0:
        os.remove("temp_uploaded_image.jpg")
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image.")

    user_face_encoding = face_recognition.face_encodings(user_image, user_face_locations)[0]

    # Perform a search on the FAISS index
    user_face_encoding = np.array([user_face_encoding], dtype=np.float32)
    distances, indices = index.search(user_face_encoding, k=5)  # Find top-5 matches

    os.remove("temp_uploaded_image.jpg")  # Clean up the temporary file

    matches = []
    for i, distance in enumerate(distances[0]):
        if distance < 0.6:  # Match threshold based on L2 distance
            similarity = (1 - distance) * 100  # Convert to percentage
            matches.append({
                "image_name": image_names[indices[0][i]],
                "similarity": f"{similarity:.2f}%"
            })

    if matches:
        return {
            "message": "Matches found!",
            "matches": matches
        }
    else:
        return {
            "message": "No good match found in the dataset.",
        }
        
