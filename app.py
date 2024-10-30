import sys
import os
import certifi
import pymongo
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import uvicorn
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

load_dotenv()
ca = certifi.where()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        
        model_dir = "final_model"
        model_file = next((f for f in os.listdir(model_dir) if f.endswith("_model.pkl")), None)
        
        if model_file is None:
            raise FileNotFoundError("No model file found ending with '_model.pkl' in the 'final_model' directory.")

        final_model_path = os.path.join(model_dir, model_file)
        network_model = load_object(final_model_path)

        y_pred = network_model.predict(df)

        label_mapping = {
            0: 'BENIGN',
            1: 'PortScan',
            2: 'DDoS',
            3: 'FTP-Patator',
            4: 'SSH-Patator'
        }

        df['predicted_column'] = [label_mapping[label] for label in y_pred]

        output_path = 'prediction_output/output.csv'
        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv(output_path, index=False)
        
        table_html = df.to_html(classes='table table-striped')
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})

    except Exception as e:
        raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
