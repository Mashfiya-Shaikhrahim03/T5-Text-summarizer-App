# Text Summarization App

A small NLP project I built to understand how text summarization works and how to use a pretrained model in a web application.

This app uses a pretrained T5 summarization model from Hugging Face to generate a shorter version of text entered by the user. I connected the model to a web interface so users can enter text and view the generated summary.

## What I worked on

- Used a pretrained T5 model from Hugging Face for text summarization.
- Cleaned the input text before passing it to the model.
- Built the Python backend to process user input and return the summary.
- Created a simple interface using HTML, CSS, and JavaScript.
- Worked with web frameworks and API-related functionality while connecting the application components.

## Tech Stack

- **Language:** Python
- **NLP Model:** T5 (Hugging Face Transformers)
- **Backend:** Flask
- **API:** FastAPI, if used in the application
- **Frontend:** HTML, CSS, JavaScript

## How it works

1. The user enters text in the web interface.
2. The application cleans the input.
3. The cleaned text is passed to the pretrained T5 model.
4. The model generates a summary.
5. The summary is displayed on the page.
