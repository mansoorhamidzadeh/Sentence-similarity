import numpy as np
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
import importlib
from mian.excute import testtargert
app = FastAPI()
router = APIRouter()
tf=testtargert()

class Item(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/result')
def predict(item:Item):
    return np.array(tf.test(item.text))
@app.get('/sync')
def sync():
    return tf.syned_dataset()

