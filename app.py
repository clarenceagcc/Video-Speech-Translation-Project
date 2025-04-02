import gradio as gr
import whisper
import tempfile
import torch
import torchaudio
import os
import subprocess
import librosa
from google.cloud import texttospeech  # Google TTS API
import collections
from phonetic_abbreviations import ABBREVIATIONS
from text_refinement import refine_transcription, translate_text
from pyAudioAnalysis import audioBasicIO
#from pyAudioAnalysis.audioFeatureExtraction import stFeatureExtraction  # Updated import
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import ShortTermFeatures
import numpy as np                                                                          

# Check if GPU is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model (cached)
model = whisper.load_model("tiny").to(device)
def detect_gender(audio_path):
    # Load the audio file
    sampling_rate, signal = audioBasicIO.read_audio_file(audio_path)
    
    # Classify gender using the pre-trained model
    result = aT.file_classification(
        audio_path,
        "pretrained_models/pyAudioAnalysis/pyAudioAnalysis/data/models/svm_rbf_speaker_male_female",
        "svm"
    )
    
    # Debug: Print the structure of the result object
    print("Result:", result)
    # Extract the predicted gender probabilities and labels
    if isinstance(result, tuple) and len(result) >= 3:
        probabilities = result[1]  # Array of probabilities
        labels = result[2]         # List of class labels
        
        # Ensure the labels are in the expected order
        if labels[0] == "Male" and labels[1] == "Female":
            male_prob = probabilities[0]
            female_prob = probabilities[1]
        else:
            raise ValueError("Unexpected class labels in result")
    else:
        raise ValueError("Unexpected result format from file_classification")
    
    # Determine the predicted gender
    predicted_gender = "female" if female_prob > male_prob else "male"
    print(f"Female Probability: {female_prob:.2f}, Male Probability: {male_prob:.2f}")
    print("Predicted Gender:", predicted_gender)
    
    return predicted_gender

# Function to get Google TTS client
def get_google_tts_client():
    client = texttospeech.TextToSpeechClient()
    return client

# Function to preprocess text for TTS
def preprocess_text(text):
    # Split the text into words
    words = text.split()
    processed_words = []
    for word in words:
        # If the word is an abbreviation (all uppercase), process it
        if word.isupper() and len(word) > 1:
            # Split the abbreviation into individual letters
            letters = list(word)
            # Replace each letter with its phonetic equivalent
            phonetic_word = " ".join([ABBREVIATIONS.get(letter.lower(), letter) for letter in letters])
            processed_words.append(phonetic_word)
        else:
            processed_words.append(word)
    return " ".join(processed_words)

# Function to extract audio from video
def extract_audio(video_path, output_audio_path, start_time=0, duration=None):
    command = [
        'ffmpeg', '-y',  # Add -y to overwrite output files automatically
        '-i', video_path, 
        '-ss', str(start_time),  # Start time for extraction
    ]
    if duration is not None:
        command.extend(['-t', str(duration)])  # Duration of extraction
    command.extend([
        '-vn', '-acodec', 'mp3', '-ar', '16000', '-ac', '1', 
        output_audio_path
    ])
    subprocess.run(command, check=True)

def split_text_into_chunks(text, max_length=5000):
    """
    Split text into smaller chunks that are below the byte size limit.
    """
    text_chunks = []
    current_chunk = ""
    for word in text.split():
        # Temporarily add the word to the current chunk
        temp_chunk = current_chunk + " " + word if current_chunk else word
        # Check if the chunk exceeds the size limit
        if len(temp_chunk.encode('utf-8')) <= max_length:
            current_chunk = temp_chunk
        else:
            # If the chunk exceeds the limit, add the current chunk and start a new one
            text_chunks.append(current_chunk)
            current_chunk = word
    if current_chunk:  # Add the last chunk
        text_chunks.append(current_chunk)
    return text_chunks


def generate_tts_audio(text, target_language, detected_gender):
    # Split the text into smaller chunks
    print(f"Detected Gender: {detected_gender}")
    text_chunks = split_text_into_chunks(text)
    tts_client = get_google_tts_client()

    audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        )

    voice_params = texttospeech.VoiceSelectionParams(
        language_code={"Chinese": "zh-CN", "English": "en-US", "Malay": "ms-MY", "Tamil": "ta-IN"}[target_language],
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE if detected_gender == "female" else texttospeech.SsmlVoiceGender.MALE
    )

    tts_audio_paths = []  # List to hold the paths of generated audio files
    
    for chunk in text_chunks:
        # Translate the chunk
        translated_chunk = translate_text(chunk, target_language)
        
        # Synthesize speech for the chunk
        synthesis_input = texttospeech.SynthesisInput(text=translated_chunk)
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice_params, audio_config=audio_config
        )
        
        # Save each chunk's audio to a separate file
        tts_audio_path = "audio_chunk_{}.wav".format(len(tts_audio_paths))
        with open(tts_audio_path, "wb") as out:
            out.write(response.audio_content)
        
        tts_audio_paths.append(tts_audio_path)
    
    return tts_audio_paths


# Function to adjust audio speed using FFmpeg
def adjust_audio_speed(input_audio_path, output_audio_path, speed_factor):
    command = [
        'ffmpeg', '-y',  # Add -y to overwrite output files automatically
        '-i', input_audio_path,
        '-filter:a', f'atempo={speed_factor}',
        output_audio_path
    ]
    subprocess.run(command, check=True)

# Function to extract video without audio
def extract_video_only(video_path, output_video_path):
    command = [
        'ffmpeg', '-y',  # Overwrite existing files
        '-i', video_path,
        '-c:v', 'copy',  # Copy the video codec without re-encoding
        '-an',  # Remove audio
        output_video_path
    ]
    subprocess.run(command, check=True)

# Function to merge video with generated TTS audio
def merge_video_audio(video_path, audio_path, output_path):
    command = [
        'ffmpeg', '-y',  # Overwrite existing files
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',  # Keep original video quality
        '-c:a', 'aac', '-b:a', '192k',  # Encode audio to AAC
        '-shortest',  # Ensure video and audio lengths match
        output_path
    ]
    subprocess.run(command, check=True)

# Gradio interface function
def process_video(video_file, target_language="en"):
    video_path = video_file

    reference_audio_path = video_path.replace(".mp4", "_reference.mp3")
    extract_audio(video_path, reference_audio_path, start_time=0, duration=15)

    audio_path = video_path.replace(".mp4", ".mp3")
    extract_audio(video_path, audio_path)

    # Extract video without audio
    video_no_audio_path = video_path.replace(".mp4", "_video.mp4")
    extract_video_only(video_path, video_no_audio_path)

    original_duration = librosa.get_duration(filename=audio_path)

    result = model.transcribe(audio_path, word_timestamps=True)

    transcription_text = []
    tts_segments = []

    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]
        
        refined_text = refine_transcription(text)  # 🔹 Apply refinement here
        
        transcription_text.append(f"[{start_time:.2f}s - {end_time:.2f}s] {refined_text}")
        tts_segments.append(refined_text)

    tts_text = " ".join(tts_segments)
    tts_text_processed = preprocess_text(tts_text)

    # Translate the transcription text
    translated_text = translate_text(tts_text_processed, target_language)

    gender = detect_gender(reference_audio_path)

    # Generate TTS audio for the translated text (now handling long texts)
    tts_audio_paths = generate_tts_audio(translated_text, target_language, gender)

    # Combine the generated audio chunks into one file
    combined_audio_path = video_path.replace(".mp4", "_combined_tts.wav")
    command = ['ffmpeg', '-y', '-i', f'concat:{"|".join(tts_audio_paths)}', '-acodec', 'pcm_s16le', '-ar', '16000', combined_audio_path]
    subprocess.run(command, check=True)

    tts_audio_duration = librosa.get_duration(filename=combined_audio_path)
    speed_factor = tts_audio_duration / original_duration

    print(f"Original Duration: {original_duration:.2f}s")
    print(f"Generated TTS Duration: {tts_audio_duration:.2f}s")
    print(f"Calculated Speed Factor: {speed_factor:.2f}")

    # Adjust the speed of TTS audio to match the original video length
    adjusted_tts_audio_path = combined_audio_path.replace(".wav", "_adjusted.wav")
    adjust_audio_speed(combined_audio_path, adjusted_tts_audio_path, speed_factor)

    # Merge the extracted video with the new TTS audio
    final_video_path = video_path.replace(".mp4", "_final.mp4")
    merge_video_audio(video_no_audio_path, adjusted_tts_audio_path, final_video_path)

    os.remove(audio_path)
    os.remove(reference_audio_path)
    os.remove(combined_audio_path)
    for tts_audio_path in tts_audio_paths:
        os.remove(tts_audio_path)


    return final_video_path, "\n".join(transcription_text), translated_text

# Gradio interface setup
iface = gr.Interface(
    fn=process_video,
    inputs=[
        gr.Video(label="Input Video"),
        gr.Dropdown(label="Target Language", choices=["Chinese", "English", "Malay", "Tamil"], value="English")
    ],
    outputs=[
        gr.Video(label="Final Video with TTS Audio"),
        gr.Textbox(label="Transcription with Timing"),
        gr.Textbox(label="Translated Text")
    ],
    title="🎥 Video Speech Transcription",
    description="Upload a video file to transcribe speech, translate it, and generate audio using Google TTS"
)


demo = iface

iface.launch()