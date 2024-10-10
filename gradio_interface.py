
import gradio as gr
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, DistilBertTokenizer, DistilBertForSequenceClassification
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models (ensure these are the fine-tuned versions)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

try:
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english').to(device)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

def speech_to_text(audio):
    try:
        # Resample audio to 16kHz for Wav2Vec2
        input_values = wav2vec2_processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(device)
        logits = wav2vec2_model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = wav2vec2_processor.batch_decode(predicted_ids)[0]
        return transcription
    except Exception as e:
        logger.error(f"Error in speech_to_text: {str(e)}")
        return f"Error in speech_to_text: {str(e)}\n{traceback.format_exc()}"

def analyze_sentiment(text):
    try:
        inputs = distilbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = distilbert_model(**inputs)
        sentiment = torch.argmax(outputs.logits, dim=1)
        sentiment_label = "Positive" if sentiment.item() == 1 else "Negative"
        return sentiment_label
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {str(e)}")
        return f"Error in analyze_sentiment: {str(e)}\n{traceback.format_exc()}"

def extract_audio_features(audio):
    try:
        # Ensure audio is in floating-point format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        pitch, _ = librosa.piptrack(y=audio, sr=16000)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=16000)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=16000)[0]
        return np.mean(pitch), tempo, np.mean(spectral_centroids)
    except Exception as e:
        logger.error(f"Error in extract_audio_features: {str(e)}")
        return f"Error in extract_audio_features: {str(e)}\n{traceback.format_exc()}", 0, 0

def process_audio(audio, sample_rate):
    try:
        logger.info(f"Processing audio with sample rate: {sample_rate}")
        # Ensure audio is in floating-point format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Resample audio to 16kHz for Wav2Vec2
        audio_16k = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=16000)

        # Speech to text
        transcription = speech_to_text(audio_16k)
        logger.info(f"Transcription: {transcription}")

        # Sentiment analysis
        sentiment = analyze_sentiment(transcription)
        logger.info(f"Sentiment: {sentiment}")

        # Extract audio features
        pitch, tempo, tone = extract_audio_features(audio)
        logger.info(f"Audio features - Pitch: {pitch}, Tempo: {tempo}, Tone: {tone}")

        return transcription, sentiment, pitch, tempo, tone
    except Exception as e:
        error_msg = f"Error in process_audio: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg, "Error", 0, 0, 0

def audio_sentiment_analyzer(audio):
    if audio is None:
        logger.warning("No audio received")
        return "No audio recorded", "N/A", None, None
    
    try:
        sr, audio = audio
        logger.info(f"Received audio with sample rate: {sr} and shape: {audio.shape}")
        
        transcription, sentiment, pitch, tempo, tone = process_audio(audio, sr)

        return transcription, sentiment, pitch, tempo, tone
    except Exception as e:
        error_msg = f"Error in audio_sentiment_analyzer: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg, error_msg, 0, 0, 0

# Create Gradio interface
iface = gr.Interface(
    fn=audio_sentiment_analyzer,
    inputs=gr.Audio(source="microphone", type="numpy"),  # Use microphone as input source
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Pitch"),
        gr.Textbox(label="Tempo"),
        gr.Textbox(label="Tone")
    ],
    title="Audio Sentiment Classifier",
    description="Record audio to analyze its sentiment and transcribe the speech.",
    allow_flagging="never"
)

def launch_interface():
    logger.info("Launching Gradio interface...")
    iface.launch(debug=True, share=True)
    logger.info("Gradio interface launched successfully.")

if __name__ == "__main__":
    launch_interface()

