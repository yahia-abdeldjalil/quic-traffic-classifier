# quic-backend/app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import FlowFeatures
from app.predictor import predict, predict_pcap

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict-pcap")
def run_predict_pcap(file: UploadFile = File(...), strategy: str = "avg"):
    print(f"Received file: {file.filename}, strategy: {strategy}")
    return predict_pcap(file, strategy)

@app.post("/predict")
def run_prediction(payload: FlowFeatures):
    print("Received payload:", payload)
    result = predict(payload.features, payload.strategy)

    return result
