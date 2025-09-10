from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import JSONResponse
import face_recognition
from datetime import datetime
import os, base64
import numpy as np

app = FastAPI(title="Absensi Wajah API (Lite)", version="1.0.0")

DATASET_PATH = "dataset"
COMPARE_PATH = "compare"
ABSENSI_FILE = "absensi.csv"
THRESHOLD = 0.6   # face_recognition pakai jarak (semakin kecil semakin mirip)


# === Load dataset wajah ===
known_encodings = []
known_names = []

for file in os.listdir(DATASET_PATH):
    path = os.path.join(DATASET_PATH, file)
    img = face_recognition.load_image_file(path)
    enc = face_recognition.face_encodings(img)
    if enc:
        known_encodings.append(enc[0])
        known_names.append(os.path.splitext(file)[0])


# === Fungsi simpan absensi ke CSV ===
def save_attendance(name):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ABSENSI_FILE, "a", encoding="utf-8") as f:
        f.write(f"{name},{now}\n")


# === Endpoint root ===
@app.get("/")
async def root():
    return {"message": "API Absensi Wajah Lite aktif 🚀"}


# === Compare via FILE ===
@app.post("/compare")
async def compare_face_file(file: UploadFile = File(...)):
    try:
        os.makedirs(COMPARE_PATH, exist_ok=True)
        file_location = os.path.join(COMPARE_PATH, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        img = face_recognition.load_image_file(file_location)
        enc = face_recognition.face_encodings(img)

        if not enc:
            return {"status": "Wajah tidak terdeteksi"}

        distances = face_recognition.face_distance(known_encodings, enc[0])
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        if best_distance <= THRESHOLD:
            name = known_names[best_idx]
            save_attendance(name)
            return {"dataset": name, "status": "COCOK", "distance": float(best_distance)}
        else:
            return {"status": "TIDAK COCOK", "distance": float(best_distance)}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# === Compare via BASE64 ===
@app.post("/compare_base64")
async def compare_face_base64(payload: dict = Body(...)):
    try:
        image_b64 = payload.get("image")
        if not image_b64:
            return JSONResponse({"error": "Base64 image tidak ditemukan"}, status_code=400)

        os.makedirs(COMPARE_PATH, exist_ok=True)
        file_location = os.path.join(COMPARE_PATH, "temp_image.jpg")

        image_bytes = base64.b64decode(image_b64)
        with open(file_location, "wb") as f:
            f.write(image_bytes)

        img = face_recognition.load_image_file(file_location)
        enc = face_recognition.face_encodings(img)

        if not enc:
            return {"status": "Wajah tidak terdeteksi"}

        distances = face_recognition.face_distance(known_encodings, enc[0])
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        if best_distance <= THRESHOLD:
            name = known_names[best_idx]
            save_attendance(name)
            return {"dataset": name, "status": "COCOK", "distance": float(best_distance)}
        else:
            return {"status": "TIDAK COCOK", "distance": float(best_distance)}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# === Attendance list ===
@app.get("/attendance")
async def get_attendance():
    if not os.path.exists(ABSENSI_FILE):
        return {"message": "Belum ada absensi"}

    with open(ABSENSI_FILE, "r", encoding="utf-8") as f:
        data = f.read().splitlines()

    return {"attendance": data}
