
import streamlit as st
import pandas as pd
import pickle
from src.analyzer import TextAuthenticityAnalyzer

# Load the trained model and analyzer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('analyzer.pkl', 'rb') as f:
    analyzer = pickle.load(f)

# App title and description
st.title('The Impostor Hunt in Text')
st.write('Detecting AI-Generated Text')

# Text input areas
text1 = st.text_area('Text 1', 'Enter the first text here...')
text2 = st.text_area('Text 2', 'Enter the second text here...')

# Analyze button
if st.button('Analyze'):
    if text1 and text2:
        # Create a dataframe with the input texts
        data = {'id': ['test_0000'], 'text1': [text1], 'text2': [text2]}
        df = pd.DataFrame(data)

        # Analyze the texts and extract features
        features = analyzer.analyze(df)

        # Predict the AI-generated text
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        # Display the result
        st.subheader('Result')
        if prediction == 1:
            st.write('**Text 1 is more likely to be AI-generated.**')
            st.write(f'Confidence: {probability[0]:.2f}')
        else:
            st.write('**Text 2 is more likely to be AI-generated.**')
            st.write(f'Confidence: {probability[1]:.2f}')
    else:
        st.write('Please enter both texts to analyze.')

