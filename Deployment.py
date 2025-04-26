# coding: utf-8

from transformers.utils import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

import os
import gradio as gr
from transformers import pipeline

pipe = pipeline("image-to-text", model="./models/Salesforce/blip-image-captioning-base")

def launch(input):
    out = pipe(input)
    return out[0]['generated_text']

iface = gr.Interface(launch, inputs=gr.Image(type='pil'), outputs="text")
iface.launch(share=True, server_port=int(os.environ['PORT1']))
iface.close()
