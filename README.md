# Audio Sentiment Classifier

## Project Overview

This project is an **Audio Sentiment Classifier** that leverages advanced natural language processing and audio processing techniques to analyze the sentiment of spoken audio. The application utilizes fine-tuned models from Hugging Face Transformers, specifically the Wav2Vec2 model for speech recognition and DistilBERT for sentiment classification.

## Key Features

- **Model Loading**: The application loads fine-tuned models:
  - **Wav2Vec2** for converting audio to text.
  - **DistilBERT** for determining the sentiment of the transcribed text.

- **Audio Processing Pipeline**:
  - **Speech-to-Text Conversion**: Converts recorded audio into textual representation.
  - **Sentiment Analysis**: Classifies the sentiment of the transcribed text as either "Positive" or "Negative."

- **Feature Extraction**: Extracts audio features, including pitch, tempo, and spectral centroid using the **Librosa** library, providing additional insights into audio characteristics.

- **Error Handling and Logging**: Implements robust error handling and logging mechanisms to track model loading, audio processing, and feature extraction, ensuring smooth application performance.

- **Gradio Interface**:
  - An interactive user interface allows users to record audio directly from the browser.
  - Displays the results, including:
    - Transcription of the speech.
    - Sentiment classification.
    - Extracted audio features (pitch, tempo, tone).

## User Experience

The application is designed for ease of use, allowing users to perform audio sentiment analysis quickly. Users can record audio through the Gradio interface, which provides clear labels for each output, enhancing understanding of the results.

## Current Progress

As of now, the **Audio Sentiment Classifier** is fully functional with real-time audio processing and analysis capabilities. The next steps will involve further fine-tuning of the models to improve performance and accuracy, as well as potential enhancements to the user interface based on feedback.

## Requirements

To run this project, you'll need the following libraries:
- `torch`
- `transformers`
- `librosa`
- `gradio`

You can install the required libraries using pip:

```bash
pip install torch transformers librosa gradio
