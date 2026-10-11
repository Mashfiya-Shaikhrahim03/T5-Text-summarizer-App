# Fast API
from fastapi import FastAPI, Request
from pydantic import BaseModel

from transformers import T5ForConditionalGeneration, T5Tokenizer

import torch
import re

from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


from fastapi import FastAPI


# initialize our fastapi app
app = FastAPI(title="Text summarizer app", description="Text summarizer using t5", version="0.1")

# model & tokenizer
model = T5ForConditionalGeneration.from_pretrained("./content/saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./content/saved_summary_model")

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else :
    device = torch.device("cpu")     

# templating
templates = Jinja2Templates(directory=".")

# input schema for dialogue => str
class dialogueInput(BaseModel):
    dialogue : str


def clean_data(text):

  text = re.sub(r"[\r\n]+", " ", text) # Lines
  text = re.sub(r"\s+", " ", text) # spaces
  text = re.sub(r"<.*?>", " ", text) # html tags

  text = text.strip().lower()
  return text

def summarize_dialogue(dialogue : str) -> str :
    dialogue = clean_data(dialogue) # clean

    # tokenize
    inputs = tokenizer(
        dialogue, 
        padding="max_length",
        max_length = 512,
        truncation=True,
        return_tensors="pt"

    ).to(device)

    with torch.no_grad():

        # generate the summary => token ids
        model.to(device)   
        targets = model.generate(

            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_length = 150,
            num_beams =  4,
            early_stopping=True
    ) 

    # decode our output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True) # EOS , SEP
    return summary

# --------------------------
# API endpoints
# --------------------------

@app.post("/summarize/")
async def summarize(dialogue_input : dialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {
        "summary" : summary
        }


# -------------------------
# Home Page
#-------------------------

@app.get("/" , response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"reques": request}

    )