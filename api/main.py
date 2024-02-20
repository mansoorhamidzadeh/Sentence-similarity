import numpy as np
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

from data_cleaning.clean import cleanSentence
from maiin.excute import testtargert

app = FastAPI()
router = APIRouter()
tf = testtargert()
clean_sent = cleanSentence()


class Item(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get('/result')
def predict(item: Item):
    text = clean_sent.clean_text(item.text)
    return np.array(tf.test(text))


@app.get('/sync')
def sync():
    return tf.syned_dataset()

# %%
