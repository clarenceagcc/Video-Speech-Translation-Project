# Video Speech Translation Project

This project implements a video speech translation pipeline using several open-source and cloud-based tools. It allows users to upload a video, extract the audio, transcribe it, refine and translate the text, generate text-to-speech audio, and embed the translated audio back into the video.

## Features

* **Video to Audio Extraction:** Uses `ffmpeg` to extract audio from uploaded video files.
* **Speech-to-Text Transcription:** Employs `whisperai` for transcribing the extracted audio.
* **Text Refinement:** Leverages `gpt-4o-mini` to improve the clarity and grammar of the transcribed text.
* **Text Translation:** Translates the refined text using `gpt-4o-mini` into English, Malay, Tamil, and Mandarin, as well as any language supported by whisperai to english.
* **Text-to-Speech (TTS):** Utilizes Google Cloud TTS for generating speech from the translated text.
* **Audio Embedding:** Integrates the generated TTS audio back into the original video using `ffmpeg`.
* **Multi-language Support:** Supports any language supported by `whisperai` as the source language, and English, Malay, Tamil, and Mandarin as target languages.

## Requirements

* Python 3.x
* `ffmpeg` installed and accessible in your system's PATH.
* Google Cloud API credentials configured for TTS.
* OpenAI API key.

## Installation

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3.  Configure Google Cloud API credentials:

    * Follow the Google Cloud documentation to set up a project and enable the Text-to-Speech API.
    * Download the service account JSON key file.
    * Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of the JSON key file.

        ```bash
        export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/google-credentials.json"
        ```

4.  Replace the placeholder OpenAI API key in `app.py` with your actual key:

    ```python
    openai.api_key = "YOUR_OPENAI_API_KEY"
    ```

## Usage

1.  Run the application:

    ```bash
    python app.py
    ```

2.  Follow the on-screen instructions to upload a video and select the target language.

## Potential Improvements

* **Performance Optimization:**
    * Explore using GPU acceleration for `whisperai` to significantly reduce processing time.
    * Implement asynchronous processing to handle different stages of the pipeline concurrently.
    * Consider using a faster transcription model, or a model that is more optimized for the local machine.
* **Translation Accuracy:**
    * Improve translation quality by refining prompts for `gpt-4o-mini` with more context and specific instructions.
    * Implement post-translation editing by allowing users to review and correct the translated text.
    * Explore using a specialized translation model, fine tuned for the target languages.
* **Audio Synchronization:**
    * Implement more sophisticated audio stretching and compression techniques to ensure better synchronization between the translated audio and the original video.
    * Implement a method to analyze the original speech rhythm, and attempt to match the tts output to the rhythm.
* **TTS Quality:**
    * Investigate alternative TTS engines with more natural-sounding voices.
    * Explore using techniques like WaveNet or similar to improve the naturalness of the google tts output.
    * Once language support is added, explore voice cloning models for a more personalized experience.
* **Subtitle Generation:**
    * Generate subtitle files(srt, vtt) of both the original transcription, and the translated text.
