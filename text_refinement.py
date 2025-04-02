import os
import re
import librosa
import numpy as np
import torch
import whisper
from pydub import AudioSegment
from openai import OpenAI


client = OpenAI(api_key = "OPEN API KEY")

# System prompt
SYSTEM_PROMPT = """You are an expert transcription editor with precise attention to detail.
Your task is to correct transcription errors while maintaining the exact word count.

CRITICAL GUIDELINES:
- Fix obvious misheard words and homophones (e.g., "they're/their/there", "to/too/two")
- Correct grammatical errors and punctuation without changing meaning
- NEVER alter technical terms, product names, or specialized vocabulary
- Maintain EXACTLY the same word count as the original
- Preserve sentence structure and speaker intent
- Pay special attention to commonly misheard phrases and number sequences
- When multiple interpretations are possible, choose the one that makes most logical sense in context
- Do not paraphrase or summarize - correct errors only
- Remove only unnecessary filler words (um, uh, you know) **when they do not impact meaning
- Do not add extra context, summaries, or explanations

Before submitting, count words to ensure the exact original count is preserved.
"""

def enhance_audio(audio_path):
    """Load and enhance audio clarity using noise reduction and volume normalization."""
    y, sr = librosa.load(audio_path, sr=16000)  # Standard ASR sampling rate
    y = librosa.effects.preemphasis(y)  # Improve speech clarity
    y = librosa.util.normalize(y)  # Normalize volume
    enhanced_path = audio_path.replace(".mp3", "_enhanced.wav")
    librosa.output.write_wav(enhanced_path, y, sr)
    return enhanced_path

def refine_transcription(text):
    # Remove common filler words **only when standalone** (avoids modifying valid words)
    filler_words = r"\b(um|uh|like|you know|please note|抱歉，我無法協助滿足該請求。)\b"
    text = re.sub(filler_words, "", text, flags=re.IGNORECASE)

    # Remove GPT disclaimers **more precisely**
    text = re.sub(r"(?i)\b(Please note that|Due to the).*?(\.|$)", "", text)

    # Call OpenAI GPT for **error correction while keeping word count**
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use GPT-4o for better transcription correction
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
    )
    
    return response.choices[0].message.content.strip()


SYSTEM_PROMPT_TRANSLATION = """You are an elite multilingual translator with specialized expertise in Chinese, Malay, and Tamil translation.
Your mission is to produce highly accurate translations with perfect handling of technical terminology.

CHINESE TRANSLATION SPECIFIC REQUIREMENTS:
- Always preserve technical terminology precisely - do not translate technical product names, model numbers, or programming terms unless there is an official Chinese equivalent
- Follow standard Chinese technical writing conventions for IT, business, and scientific domains
- For technical manuals or documentation, use formal Simplified Chinese (简体中文) unless otherwise specified
- Numerals must be translated with 100% accuracy - verify all numbers match the source text
- Use proper Chinese punctuation (，。：；""（）) not Western punctuation when the output is in Chinese
- Handle measure words (量词) correctly according to Chinese grammar rules
- Technical abbreviations (API, SDK, UI, etc.) should remain in Latin characters
- Website URLs, file paths, and code snippets must remain unchanged
- Maintain the exact hierarchical structure and formatting of the original
- Pay special attention to proper nouns that should be transliterated according to standard conventions

- For Malay: Follow standard Malay technical writing conventions; preserve technical terms in English; use proper Malay sentence structure
- For Tamil: Use formal Tamil for technical content; maintain proper Tamil grammar and sentence structure; preserve technical terms in English 


Common technical translation errors to avoid:
- Incorrect handling of technical jargon
- Unnatural word order in technical descriptions
- Inconsistent terminology within the same document
- Literal translations of idioms that lose meaning
- Incorrect formatting of numbers (dates, times, currencies, measurements)

After completing your translation, verify all technical terms and numerals match the source.
- All technical terms remain in English (e.g., API, SDK, HTTP)**
- Numerals are 100% correct**
- Formatting matches the original structure**
"""


def translate_text(text, target_language="Chinese"):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Use GPT-4o for better translation quality
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TRANSLATION},
            {"role": "user", "content": f"Translate the following text into {target_language} without introductory phrases:\n\n{text}"}
        ]
    )
    
    return response.choices[0].message.content.strip()