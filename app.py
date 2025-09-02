import streamlit as st
import pandas as pd
import numpy as np
import pickle
from src.analyzer import TextAuthenticityAnalyzer
from src.classifier import TextAuthenticityClassifier

st.set_page_config(page_title="The Impostor Hunt in Text", page_icon="🕵️‍♀️", layout="wide")

@st.cache(allow_output_mutation=True)
def load_model():
    # Load the trained model and analyzer
    # Note: You will need to train the model and save it as a pickle file first
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("analyzer.pkl", "rb") as f:
            analyzer = pickle.load(f)
    except FileNotFoundError:
        st.error("Model not found. Please train the model first.")
        return None, None
    return model, analyzer

def main():
    st.title("🕵️‍♀️ The Impostor Hunt in Text")
    st.markdown("This application uses a machine learning model to distinguish between human-written and AI-generated text. Enter a text below to see the prediction.")

    model, analyzer = load_model()

    if model and analyzer:
        text_input = st.text_area("Enter your text here:", height=200)

        if st.button("Analyze Text"):
            if text_input:
                # Create a dummy dataframe for the analyzer
                data = {'text_1': [text_input], 'text_2': [""], 'real_text_id': [1]}
                df = pd.DataFrame(data)

                # Extract features
                features_df = analyzer.analyze_dataset(df)

                # Make prediction
                prediction = model.predict(features_df)
                prediction_proba = model.predict_proba(features_df)

                # Display result
                st.subheader("Analysis Result")
                if prediction[0] == 1:
                    st.success("This text is likely to be **human-written**.")
                else:
                    st.error("This text is likely to be **AI-generated**.")
                
                st.write(f"Confidence Score: {prediction_proba[0][prediction[0]]:.2f}")

            else:
                st.warning("Please enter a text to analyze.")

if __name__ == "__main__":
    main()