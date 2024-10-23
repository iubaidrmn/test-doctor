import streamlit as st
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Load the tokenizer and model
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Streamlit app
st.title('Health Symptom Checker Chatbot')
st.write("Enter your symptoms, and the chatbot will suggest possible reasons for your symptoms.")

# User input
input_text = st.text_input('Describe your symptoms (e.g., headache, fever):')

# Add a [MASK] token for the symptom we want to predict
masked_input = f"Patient presents with {input_text} and [MASK]."

if st.button('Get Suggestions'):
    # Tokenize the input text with the masked token
    inputs = tokenizer(masked_input, return_tensors="pt")

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predictions for the masked token
    predictions = outputs.logits

    # Find the index of the mask token
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    # Get the top 5 predictions for the [MASK]
    top_k = 5
    mask_token_logits = predictions[0, mask_token_index, :]
    top_k_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()

    # Decode the top_k tokens
    predicted_tokens = [tokenizer.decode([token]) for token in top_k_tokens]

    # Display the suggestions
    st.write(f"Possible symptoms or conditions related to your input: {', '.join(predicted_tokens)}")
