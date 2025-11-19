


# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import io
# from PIL import Image


# # Load the trained model
# model = load_model("V:/projects/Alzheimers_classification/model/cnn_model.h5")

# # Class labels
# class_labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# app = FastAPI()
# from fastapi.middleware.cors import CORSMiddleware

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all domains (change to specific domain in production)
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all HTTP methods
#     allow_headers=["*"],  # Allows all headers
# )


# # Function to preprocess the image
# def preprocess_image(img):
#     img = img.resize((128, 128))  # Resize to match model input
#     img = np.array(img) / 255.0   # Normalize pixel values
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# # API endpoint to upload and predict image
# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Read and open image
#         img = Image.open(io.BytesIO(await file.read()))
#         img = img.convert("RGB")  # Ensure RGB format

#         # Preprocess image
#         img = preprocess_image(img)

#         # Perform model prediction
#         predictions = model.predict(img)
#         predicted_class = np.argmax(predictions, axis=1)[0]
#         confidence = float(np.max(predictions))

#         return JSONResponse(content={
#             "filename": file.filename,
#             "prediction": class_labels[predicted_class],
#             "confidence": confidence
#         })

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=400)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import os
import boto3
import tempfile
from fastapi.middleware.cors import CORSMiddleware


# -------------------------------------------------------
#  AWS MODEL LOADING SECTION (Safe + Optional)
# -------------------------------------------------------

# Environment variable (set in Elastic Beanstalk console):
# MODEL_S3_URI = "s3://your-bucket-name/cnn_model.h5"
MODEL_S3_URI = os.environ.get("MODEL_S3_URI")  # None if not set


def download_model_from_s3(s3_uri):
    """
    Download model from S3 to a temporary local file
    Only used if MODEL_S3_URI is defined.
    """
    s3 = boto3.client("s3")

    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI format")

    _, rest = s3_uri.split("s3://", 1)
    bucket, key = rest.split("/", 1)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    s3.download_fileobj(bucket, key, temp_file)
    temp_file.close()

    return temp_file.name


# -------------------------------------------------------
#  Decide model load method
# -------------------------------------------------------

if MODEL_S3_URI:
    print("📌 Loading model from S3:", MODEL_S3_URI)
    model_path = download_model_from_s3(MODEL_S3_URI)
else:
    print("📌 Loading model from LOCAL path")
    model_path = "V:/projects/AlzheimersDiseaseClassification/model/cnn_model.h5"  # your original path


# -------------------------------------------------------
#  Load the trained model (same logic as before)
# -------------------------------------------------------

model = load_model(model_path)

# Class labels (UNCHANGED)
class_labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]


# -------------------------------------------------------
#  FastAPI + CORS (UNCHANGED)
# -------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------
#  Preprocessing function (UNCHANGED)
# -------------------------------------------------------

def preprocess_image(img):
    img = img.resize((128, 128))  # Resize to match model input
    img = np.array(img) / 255.0   # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# -------------------------------------------------------
#  PREDICTION API (UNCHANGED)
# -------------------------------------------------------

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
        img = img.convert("RGB")

        img = preprocess_image(img)

        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))

        return JSONResponse(content={
            "filename": file.filename,
            "prediction": class_labels[predicted_class],
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# -------------------------------------------------------
#  LOCAL DEV RUN (UNCHANGED)
# -------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


