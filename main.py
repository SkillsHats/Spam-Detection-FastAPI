import joblib

from fastapi import FastAPI

from utils import classify_message
from model import Message

app = FastAPI()

model = joblib.load('models/spam_classifier.joblib')


@app.get('/')
def get_root():
	return {'message': 'Welcome to the spam detection API'}


@app.get('/spam_detection_query/')
async def detect_spam_query(message: str):
	return classify_message(model, message)


@app.get('/spam_detection_path/{message}')
async def detect_spam_path(message: str):
	return classify_message(model, message)


@app.post('/spam_detection/')
async def detect_spam_message(message: Message):
	return classify_message(model, message.messageText)