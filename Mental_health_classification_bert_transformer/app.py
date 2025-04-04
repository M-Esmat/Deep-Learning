import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle



# Load the model, tokenizer, and label encoder
MODEL_PATH = "Mental_health_classification_bert_transformer/mental_status_bert_v2"
TOKENIZER_PATH = "Mental_health_classification_bert_transformer/mental_status_bert_v2"
LABEL_ENCODER_PATH = "Mental_health_classification_bert_transformer/label_encoder.pkl"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
label_encoder = pickle.load(open(LABEL_ENCODER_PATH, 'rb'))


# Function for prediction
def detect_mental_health(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    return label_encoder.inverse_transform([predicted_class.item()])[0], confidence.item()

# Streamlit UI
st.set_page_config(page_title="Mental Health Detector", page_icon="🧠", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Mental Health Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555;'>Enter a sentence below to analyze your mental health state.</h4>", unsafe_allow_html=True)

st.write("")
st.write("")

# Input field
user_input = st.text_area("✍️ Type your sentence here:", "", height=150)

# Button styling
st.markdown("""
    <style>
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 24px;
            border-radius: 10px;
            border: none;
            transition: 0.3s;
        }
        div.stButton > button:first-child:hover {
            background-color: #388E3C;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

if st.button("🔍 Analyze"):
    if user_input.strip():
        prediction, confidence = detect_mental_health(user_input)

        # Emoji mapping
        emoji_dict = {
            "Anxiety": "😟",
            "Bipolar": "🔄",
            "Depression": "😞",
            "Normal": "😊",
            "Personality disorder": "🎭",
            "Stress": "😰",
            "Suicidal": "💔"
        }
        emoji = emoji_dict.get(prediction, "❓")

        # Display the prediction
        st.success(f"### {emoji} Predicted Mental Health State: {prediction}")
        st.write(f"🧐 Confidence Score: **{confidence*100:.2f}%**")
    else:
        st.warning("⚠️ Please enter a sentence to analyze.")

# Footer
st.markdown("""
    <br>
    <hr>
    <p style="text-align:center;">🤖 Built with ❤️ using <b>PyTorch & Streamlit</b></p>
""", unsafe_allow_html=True)