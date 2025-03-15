# CHETGPT: Kinda works... but results are not very accurate.

# CHATGPT

## 🎵 Overview
SONIX ANALYSIS V1A is an advanced **audio analysis and classification tool** that processes audio files to extract **musical, acoustic, and textual features**. It integrates multiple machine learning techniques, audio processing methods, and natural language processing (NLP) tools to analyze **music, speech, and textual content** from audio files.

## 🚀 Features
- **🎧 Audio Feature Extraction**: Analyzes energy, tempo, pitch, loudness, and spectral characteristics.
- **📝 Transcription & NLP**: Converts speech to text and applies sentiment analysis, named entity recognition, and summarization.
- **🔊 Music Genre Classification**: Uses **fuzzy logic-based classification** to determine probable music genres.
- **📊 Acoustic Properties Analysis**: Measures **acousticness, danceability, loudness, instrumentalness, and valence**.
- **🎛️ Audio Processing**: Separates vocals and accompaniment using Spleeter and extracts harmonic/percussive features.
- **🔎 Spotify API Integration**: Matches extracted audio features with **Spotify's track recommendation system**.
- **📂 Data Export**: Stores processed insights into an Excel-based **Sonic Analysis Database**.

## 📦 Dependencies
Ensure you have the following Python packages installed:
```bash
pip install numpy pandas librosa nltk speechrecognition spotipy requests networkx scipy pywt openai
```

## 🛠 How to Use
1. **Prepare Your Audio Files**: Place `.wav` or `.mp3` files in the designated folder.
2. **Run the Script**:
   ```bash
   python SONIX_ANALYSIS_V1A.py
   ```
3. **Follow the Interactive Labeling Process**: Assign metadata to each file (genre, artist, language, etc.).
4. **View Results**: The processed data is stored in `Sonic Analysis Database.xlsx`.

## 🎯 Key Functionalities
### 🎤 **Speech-to-Text & NLP**
- **Transcribes Speech** using Google Web Speech API
- **Named Entity Recognition (NER)** to extract key topics
- **Sentiment Analysis** with Vader Lexicon
- **Summarization** using token-based frequency scoring

### 🎼 **Music Feature Extraction**
- **Tempo & Rhythm Detection**
- **Harmonic vs Percussive Signal Separation**
- **Spectral Analysis & Frequency Mapping**
- **Energy & Loudness Classification**

### 🎶 **Music Genre Classification (Fuzzy Logic)**
- Uses **tempo, energy, loudness, and danceability** to assign probabilities for various music genres
- Aligns extracted insights with **Spotify genre mapping**

### 🎵 **Spotify API Integration**
- Extracts **Spotify track features** (acousticness, valence, tempo, etc.)
- Recommends similar tracks based on analyzed features

## 📁 File Outputs
- **`Sonic Analysis Database.xlsx`** → Stores all extracted insights and labeled metadata
- **`transcription.txt`** → Stores text transcriptions of speech-containing files
- **`storyboard.jpg`** → Visual representation of detected video scenes (if applicable)

## 📌 Next Steps
- Enhance accuracy of **fuzzy genre classification**
- Improve **voice and instrument separation techniques**
- Expand Spotify API integration for **playlist generation**

## 🤝 Contributing
Feel free to contribute by submitting issues or pull requests to improve the feature set!

---
🎵 *Bringing AI-powered analysis to music and audio processing!*
