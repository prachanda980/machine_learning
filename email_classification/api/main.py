from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.predict import predict_email

app = FastAPI()

class EmailRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: EmailRequest):
    label = predict_email(request.text)
    return {"spam": bool(label)}
