from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2

app = FastAPI()

# Cấu hình CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình model
IMAGE_SIZE = 256


# Nếu model dùng Lambda layer resize
custom_objects = {
    "<lambda>": lambda x: tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
}

try:
    MODEL = tf.keras.models.load_model("saved_models/1", custom_objects=custom_objects)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

CLASS_NAMES = ["Héo Lá Sớm", "Thối Lá Muộn", "Khỏe Mạnh"]

BLUR_THRESHOLD = 100.0  # Ngưỡng để xác định ảnh mờ
def enhance_image_if_blurry(image_np: np.ndarray, sharpness: float, threshold: float) -> np.ndarray:
    if sharpness < threshold:
        # Làm sắc nét ảnh bằng kernel convolution
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        image_np = cv2.filter2D(image_np, -1, kernel)

        # Tăng độ tương phản và độ sáng nhẹ
        image_np = cv2.convertScaleAbs(image_np, alpha=1.2, beta=15)  # alpha: tương phản, beta: sáng

    return image_np

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()

        # Đọc ảnh xám để kiểm tra độ sắc nét
        file_bytes = np.asarray(bytearray(data), dtype=np.uint8)
        img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            return {"error": "Không thể đọc ảnh."}

        laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < BLUR_THRESHOLD

        # Đọc lại ảnh RGB và resize
        image = Image.open(BytesIO(data)).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image_np = np.array(image)

        # Nếu ảnh mờ, xử lý ảnh
        image_np = enhance_image_if_blurry(image_np, laplacian_var, BLUR_THRESHOLD)

        # Chuẩn bị cho dự đoán
        img_batch = np.expand_dims(image_np, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            "class": predicted_class,
            "confidence": confidence,
            "sharpness": laplacian_var,
            "is_blurry": bool(is_blurry),  # <-- Ép kiểu ở đây
            "message": "Đã xử lý ảnh mờ trước khi dự đoán." if is_blurry else "Ảnh rõ, dự đoán trực tiếp."
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
