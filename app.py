import gradio as gr
import torch
import plotly.graph_objects as go
import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

from optimized_clinical_bert import (
    ModelConfig,
    OptimizedHierarchicalClinicalBERT
)


#!/usr/bin/env python3
"""
Inference script for Optimized Hierarchical ClinicalBERT
========================================================
Returns disease names instead of numeric labels
"""


# -----------------------------
# Label mappings (IMPORTANT)
# -----------------------------
PARENT_ID2LABEL = {
    0: "general pathological conditions",
    1: "specific disease"
}

CHILD_ID2LABEL = {
    0: "neoplasms",
    1: "digestive system diseases",
    2: "nervous system diseases",
    3: "cardiovascular diseases"
}


# -----------------------------
# Load Model
# -----------------------------
def load_model(
    model_path: str,
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    device: str = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = ModelConfig(
        model_name=model_name,
        use_lora=True,
        use_gradient_checkpointing=False  # disable for inference
    )

    model = OptimizedHierarchicalClinicalBERT(config)
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer, device


# -----------------------------
# Single Text Inference
# -----------------------------
@torch.no_grad()
def predict_single(
    text: str,
    model,
    tokenizer,
    device,
    max_len: int = 256
) -> Dict:
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    outputs = model(input_ids, attention_mask)

    # Full softmax probabilities
    parent_probs = torch.softmax(outputs["parent_logits"], dim=-1).squeeze(0)  # (2,)
    child_probs = torch.softmax(outputs["child_logits"], dim=-1).squeeze(0)    # (4,)

    parent_pred = parent_probs.argmax().item()
    child_pred = child_probs.argmax().item()
    parent_conf = parent_probs.max().item()


    # Hierarchical decoding
    if parent_conf >= 0.8:  # general
        predicted_disease = PARENT_ID2LABEL[0]
        child_label = None
    else:
        predicted_disease = CHILD_ID2LABEL[child_pred]
        child_label = CHILD_ID2LABEL[child_pred]

    # Return ALL probabilities as dicts
    parent_probs_dict = {label: float(prob) for label, prob in zip(PARENT_ID2LABEL.values(), parent_probs.tolist())}
    child_probs_dict = {label: float(prob) for label, prob in zip(CHILD_ID2LABEL.values(), child_probs.tolist())}

    return {
        "text": text,
        "predicted_disease": predicted_disease,
        "parent_label": PARENT_ID2LABEL[parent_pred],
        "parent_probs": parent_probs_dict,
        "child_label": child_label,
        "child_probs": child_probs_dict
    }




# -----------------------------
# Load model once
# -----------------------------
MODEL_PATH = "./results/best_model.safetensors"
model, tokenizer, device = load_model(MODEL_PATH)


# -----------------------------
# Function to create bar plots
# -----------------------------
def plot_probs(prob_dict, title="Probabilities"):
    labels = list(prob_dict.keys())
    values = [v if v is not None else 0 for v in prob_dict.values()]
    
    fig = go.Figure([go.Bar(x=labels, y=values, text=[round(v,3) for v in values], textposition='auto')])
    fig.update_layout(title_text=title, yaxis=dict(range=[0,1]), margin=dict(t=40, b=20))
    return fig


# -----------------------------
# Prediction function for Gradio
# -----------------------------
def classify_text(text: str):
    result = predict_single(text, model, tokenizer, device)

    # Parent and child probability dicts
    parent_display = {label: round(prob,3) for label, prob in result["parent_probs"].items()}

    if result["child_label"] is None:
        child_display = {label: 0 for label in CHILD_ID2LABEL.values()}
    else:
        child_display = {label: round(prob,3) for label, prob in result["child_probs"].items()}

    # Build plots
    parent_plot = plot_probs(parent_display, title="Parent Probabilities")
    child_plot = plot_probs(child_display, title="Child Probabilities")

    return result["predicted_disease"], parent_plot, child_plot


# -----------------------------
# Gradio Interface
# -----------------------------
iface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=10, placeholder="Enter medical abstract here..."),
    outputs=[
        gr.Label(label="Predicted Disease"),
        gr.Plot(label="Parent Probabilities"),
        gr.Plot(label="Child Probabilities")
    ],
    title="Hierarchical ClinicalBERT Disease Classifier",
    description="Enter a medical abstract to get predicted disease and probability plots for parent and child disease categories."
)

if __name__ == "__main__":
    iface.launch()