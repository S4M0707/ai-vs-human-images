import os
import pandas as pd
from sklearn.model_selection import train_test_split

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from utils.dataset_loader import DatasetLoader
from utils.predictor import ModelPredictor

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost",
    "http://localhost:8000",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dataset_path = "data"
data_loader = DatasetLoader(dataset_path,  img_size=(224, 224), batch_size=16)
predictor = ModelPredictor("model/EfficientNetB0.keras")

@app.post('/upload')
async def upload(imgFiles: list[UploadFile] = File(...)):
    imgPaths = []

    for imgFile in imgFiles:
        imgPath = os.path.join('user_files', f'{imgFile.filename}')
        with open(imgPath, 'wb') as f:
            f.write(await imgFile.read())
        imgPaths.append(imgPath)

    return {'img_paths': imgPaths}


@app.post('/predict')
async def predict(imgPaths: list[str]):
    try:
        dataset = data_loader.prepare_image_paths(imgPaths)
        predictions = predictor.predict(dataset)

        result = []
        for path, pred in zip(imgPaths, predictions.tolist()):
            result.append({
                "filename": os.path.basename(path),
                "prediction": pred
            })

        for path in imgPaths:
            if os.path.exists(path):
                os.remove(path)

        return {"predictions": result}

    except Exception as e:
        for path in imgPaths:
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=f"Error happened: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)