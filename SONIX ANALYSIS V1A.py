import json
import os
import re
import subprocess
import time
import librosa
import numpy as np
import hashlib
import pandas as pd
import nltk
import speech_recognition as sr
import spotipy
import requests
import urllib.parse
import networkx as nx
import pywt
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from scipy.stats import entropy
from scipy.stats import skew,kurtosis

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from datetime import datetime
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.tree import Tree
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

indonesian_stop_words = [
    "ada", "adalah", "adanya", "adapun", "agak", "agaknya", "agar", "akan", "akankah", "akhir", "akhiri", "akhirnya", "aku", "akulah",
    "amat", "amatlah", "anda", "andalah", "antar", "antara", "antaranya", "apa", "apaan", "apabila", "apakah", "apalagi", "apatah",
    "artinya", "asal", "asalkan", "atas", "atau", "ataukah", "ataupun", "awal", "awalnya", "bagai", "bagaikan", "bagaimana", "bagaimanakah",
    "bagaimanapun", "bagi", "bagian", "bahkan", "bahwa", "bahwasanya", "baik", "bakal", "bakalan", "balik", "banyak", "bapak", "baru",
    "bawah", "beberapa", "begini", "beginian", "beginikah", "beginilah", "begino", "begitu", "begitukah", "begitulah", "begitupun", "bekerja",
    "belakang", "belakangan", "belum", "belumlah", "benar", "benarkah", "benarlah", "berada", "berakhir", "berakhirlah", "berakhirnya",
    "berapa", "berapakah", "berapalah", "berapapun", "berarti", "berawal", "berbagai", "berdatangan", "beri", "berikan", "berikut", "berikutnya",
    "berjumlah", "berkali-kali", "berkata", "berkehendak", "berkeinginan", "berkenaan", "berlainan", "berlalu", "berlangsung", "berlebihan",
    "bermacam", "bermacam-macam", "bermaksud", "bermula", "bersama", "bersama-sama", "bersiap", "bersiap-siap", "bertanya", "bertanya-tanya",
    "berturut", "berturut-turut", "bertutur", "berujar", "berupa", "besar", "betul", "betulkah", "biasa", "biasanya", "bila", "bilakah",
    "bisa", "bisakah", "boleh", "bolehkah", "bolehlah", "buat", "bukan", "bukankah", "bukanlah", "bukannya", "bulan", "bung", "cara", "caranya",
    "cukup", "cukupkah", "cukuplah", "cuma", "dahulu", "dalam", "dan", "dapat", "dari", "daripada", "datang", "dekat", "demi", "demikian",
    "demikianlah", "dengan", "depan", "di", "dia", "diakhiri", "diakhirinya", "dialah", "diantara", "diantaranya", "diberi", "diberikan",
    "diberikannya", "dibuat", "dibuatnya", "didapat", "didatangkan", "digunakan", "diibaratkan", "diibaratkannya", "diingat", "diingatkan",
    "diinginkan", "dijawab", "dijelaskan", "dijelaskannya", "dikarenakan", "dikatakan", "dikatakannya", "dikerjakan", "diketahui", "diketahuinya",
    "dikira", "dilakukan", "dilalui", "dilihat", "dimaksud", "dimaksudkan", "dimaksudkannya", "dimaksudnya", "diminta", "dimintai", "dimisalkan",
    "dimulai", "dimulailah", "dimulainya", "dimungkinkan", "dini", "dipastikan", "diperbuat", "diperbuatnya", "dipergunakan", "diperkirakan",
    "diperlihatkan", "diperlukan", "diperlukannya", "dipersoalkan", "dipertanyakan", "dipunyai", "diri", "dirinya", "disampaikan", "disebut",
    "disebutkan", "disebutkannya", "disini", "disinilah", "ditambahkan", "ditandaskan", "ditanya", "ditanyai", "ditanyakan", "ditegaskan",
    "ditujukan", "ditunjuk", "ditunjuki", "ditunjukkan", "ditunjukkannya", "ditunjuknya", "dituturkan", "dituturkannya", "diucapkan", "diucapkannya",
    "diungkapkan", "dong", "dua", "dulu", "empat", "enggak", "enggaknya", "entah", "entahlah", "guna", "gunakan", "hal", "hampir", "hanya", "hanyalah",
    "hari", "harus", "haruslah", "harusnya", "hendak", "hendaklah", "hendaknya", "hingga", "ia", "ialah", "ibarat", "ibaratkan", "ibaratnya", "ibu",
    "ikut", "ingat", "ingat-ingat", "ingin", "inginkah", "inginkan", "ini", "inikah", "inilah", "itu", "itukah", "itulah", "jadi", "jadilah", "jadinya",
    "jangan", "jangankan", "janganlah", "jauh", "jawab", "jawaban", "jawabnya", "jelas", "jelaskan", "jelaslah", "jelasnya", "jika", "jikalau", "juga",
    "jumlah", "jumlahnya", "justru", "kala", "kalau", "kalaulah", "kalaupun", "kalian", "kami", "kamilah", "kamu", "kamulah", "kan", "kapan", "kapankah",
    "kapanpun", "karena", "karenanya", "kasus", "kata", "katakan", "katakanlah", "katanya", "ke", "keadaan", "kebetulan", "kecil", "kedua", "keduanya",
    "keinginan", "kelamaan", "kelihatan", "kelihatannya", "kelima", "keluar", "kembali", "kemudian", "kemudianlah", "kemungkinan", "kemungkinannya",
    "kenapa", "kepada", "kepadanya", "kesampaian", "keseluruhan", "keseluruhannya", "keterlaluan", "ketika", "khususnya", "kini", "kinilah", "kira",
    "kira-kira", "kiranya", "kita", "kitalah", "kok", "kurang", "lagi", "lagian", "lah", "lain", "lainnya", "lalu", "lama", "lamanya", "lanjut", "lanjutnya",
    "lebih", "lewat", "lima", "luar", "macam", "maka", "makanya", "makin", "malah", "malahan", "mampu", "mampukah", "mana", "manakala", "manalagi",
    "masa", "masalah", "masalahnya", "masih", "masihkah", "masing", "masing-masing", "mau", "maupun", "melainkan", "melakukan", "melalui", "melihat",
    "melihatnya", "memang", "memastikan", "memberi", "memberikan", "membuat", "memerlukan", "memihak", "meminta", "memintakan", "memisalkan", "memperbuat",
    "mempergunakan", "memperkirakan", "memperlihatkan", "mempersiapkan", "mempersoalkan", "mempertanyakan", "mempunyai", "memulai", "memungkinkan",
    "menaiki", "menambahkan", "menandaskan", "menanti", "menanti-nanti", "menantikan", "menanya", "menanyai", "menanyakan", "mendapat", "mendapatkan",
    "mendatang", "mendatangi", "mendatangkan", "menegaskan", "mengakhiri", "mengapa", "mengatakan", "mengatakannya", "mengenai", "mengerjakan", "mengetahui",
    "menggunakan", "menghendaki", "mengibaratkan", "mengibaratkannya", "mengingat", "mengingatkan", "menginginkan", "mengira", "mengucapkan", "mengucapkannya",
    "mengungkapkan", "menjadi", "menjawab", "menjelaskan", "menuju", "menunjuk", "menunjuki", "menunjukkan", "menunjuknya", "menurut", "menuturkan",
    "menyampaikan", "menyangkut", "menyatakan", "menyebutkan", "menyeluruh", "menyiapkan", "merasa", "mereka", "merekalah", "merupakan", "meski", "meskipun",
    "meyakini", "meyakinkan", "minta", "mirip", "misal", "misalkan", "misalnya", "mula", "mulai", "mulailah", "mulanya", "mungkin", "mungkinkah", "nah",
    "naik", "namun", "nanti", "nantinya", "nyaris", "nyatanya", "oleh", "olehnya", "pada", "padahal", "padanya", "pak", "paling", "panjang", "pantas",
    "para", "pasti", "pastilah", "penting", "pentingnya", "per", "percuma", "perlu", "perlukah", "perlunya", "pernah", "persoalan", "pertama", "pertama-tama",
    "pertanyaan", "pertanyakan", "pihak", "pihaknya", "pukul", "pula", "pun", "punya", "rasa", "rasanya", "rata", "rupanya", "saat", "saatnya", "saja", "sajalah",
    "saling", "sama", "sama-sama", "sambil", "sampai", "sampai-sampai", "sampaikan", "sana", "sangat", "sangatlah", "satu", "saya", "sayalah", "se", "sebab",
    "sebabnya", "sebagai", "sebagaimana", "sebagainya", "sebagian", "sebaik", "sebaik-baiknya", "sebaiknya", "sebaliknya", "sebanyak", "sebegini", "sebeginian",
    "sebeginikah", "sebeginilah", "sebegitu", "sebelum", "sebelumnya", "sebenarnya", "seberapa", "sebesar", "sebetulnya", "sebisanya", "sebuah", "sebut",
    "sebutlah", "sebutnya", "secara", "secukupnya", "sedang", "sedangkan", "sedemikian", "sedikit", "sedikitnya", "seenaknya", "segala", "segalanya", "segera",
    "seharusnya", "sehingga", "seingat", "sejak", "sejauh", "sejenak", "sejumlah", "sekadar", "sekadarnya", "sekali", "sekali-kali", "sekalian", "sekaligus",
    "sekalipun", "sekarang", "sekarang", "sekecil", "seketika", "sekiranya", "sekitar", "sekitarnya", "sekurang-kurangnya", "sekurangnya", "sela", "selain",
    "selaku", "selalu", "selamanya", "selanjutnya", "seluruh", "seluruhnya", "semacam", "semakin", "semampu", "semampunya", "semasa", "semasih", "semata",
    "semata-mata", "semaunya", "sementara", "semisal", "semisalnya", "sempat", "semua", "semuanya", "semula", "sendiri", "sendirian", "sendirinya", "seolah",
    "seolah-olah", "seorang", "sepanjang", "sepantasnya", "sepantasnyalah", "seperlunya", "seperti", "sepertinya", "sepihak", "sering", "seringnya", "serta",
    "serupa", "sesaat", "sesama", "sesampai", "sesegera", "sesekali", "seseorang", "sesuatu", "sesuatunya", "sesudah", "sesudahnya", "setelah", "setempat",
    "setengah", "seterusnya", "setiap", "setiba", "setibanya", "setidak-tidaknya", "setidaknya", "setinggi", "seusai", "sewaktu", "siap", "siapa", "siapakah",
    "siapapun", "sini", "sinilah", "soal", "soalnya", "suatu", "sudah", "sudahkah", "sudahlah", "supaya", "tadi", "tadinya", "tahu", "tahun", "tak", "tambah",
    "tambahnya", "tampak", "tampaknya", "tandas", "tandasnya", "tanpa", "tanya", "tanyakan", "tanyanya", "tapi", "tegas", "tegasnya", "telah", "tempat", "tengah",
    "tentang", "tentu", "tentulah", "tentunya", "tepat", "terakhir", "terasa", "terbanyak", "terdahulu", "terdapat", "terdiri", "terhadap", "terhadapnya",
    "teringat", "teringat-ingat", "terjadi", "terjadilah", "terjadinya", "terkira", "terlalu", "terlebih", "terlihat", "termasuk", "ternyata", "tersampaikan",
    "tersebut", "tersebutlah", "tertentu", "tertuju", "terus", "terutama", "tetap", "tetapi", "tiap", "tiba", "tiba-tiba", "tidak", "tidakkah", "tidaklah",
    "tiga", "tinggi", "toh", "tunjuk", "turut", "tutur", "tuturnya", "ucap", "ucapnya", "ujar", "ujarnya", "umum", "umumnya", "ungkap", "ungkapnya", "untuk",
    "usah", "usai", "waduh", "wah", "wahai", "waktu", "waktunya", "walau", "walaupun", "wong", "yaitu", "yakin", "yakni", "yang"
]

stop_words = set(indonesian_stop_words)

INDUSTRY_OPTIONS = {
    1: ("Consumer Packaged Goods (CPG)", "Includes food, beverage, toiletries, cosmetics, and household cleaning product brands."),
    2: ("Automotive", "Car and motorcycle manufacturers."),
    3: ("Telecommunications", "Phone companies, internet service providers, etc."),
    4: ("Technology and Electronics", "Companies promoting smartphones, computers, wearables, etc."),
    5: ("Entertainment", "Movie studios, music labels, video game developers, streaming services."),
    6: ("Financial Services", "Banks, insurance companies, investment firms."),
    7: ("Retail", "Online and brick-and-mortar retailers, from fashion to electronics."),
    8: ("Pharmaceuticals", "Drug companies advertising medications."),
    9: ("Travel and Tourism", "Airlines, hotel chains, travel agencies, tourism boards."),
    10: ("Real Estate", "Developers promoting housing projects, commercial spaces."),
    11: ("Education", "Universities, colleges, online course platforms."),
    12: ("Non-profits and Social Causes", "NGOs and social organizations campaigning for specific causes."),
    13: ("Other", "Any other category not listed above.")
}

INDUSTRY_SUBCATEGORIES = {
    "Consumer Packaged Goods (CPG)": [
        "Food and Beverages: Fast food, snacks, soft drinks, and alcoholic beverages.",
        "Cosmetics and Personal Care: Skincare, haircare, and beauty products.",
        "Apparel and Footwear: Clothing, shoes, and accessories.",
        "Electronics: Smartphones, laptops, and other consumer electronics.",
        "Toys and Games: Video games, board games, and children's toys."
    ],
    "Automotive": [
        "Cars and Trucks: New models, used vehicles, and vehicle services.",
        "Public Transport: Airlines, train services, and bus lines.",
        "Automotive Accessories: Tires, oils, and other maintenance products."
    ],
    "Telecommunications": [
        "Mobile Services: Mobile plans, prepaid cards, and mobile devices.",
        "Internet Services: Broadband, fiber optics, and internet plans.",
        "Cable and Satellite TV: TV packages, satellite dishes, and related equipment."
    ],
    "Technology and Electronics": [
        "Computing: Desktops, laptops, and accessories.",
        "Mobile Devices: Smartphones, tablets, and accessories.",
        "Home Electronics: TVs, audio systems, and smart home devices."
    ],
    "Entertainment": [
        "Streaming Services: Netflix, Hulu, and other platforms.",
        "Books and Magazines: Bestsellers, subscriptions, and e-books.",
        "Movies and Music: New releases, concerts, and festivals."
    ],
    "Financial Services": [
        "Banking: Checking accounts, savings accounts, and loans.",
        "Insurance: Health, car, and life insurance.",
        "Investment: Stocks, bonds, and mutual funds."
    ],
    "Retail": [
        "Online Retail: E-commerce websites and online marketplaces.",
        "Physical Stores: Department stores, specialty shops, and supermarkets.",
        "Fashion and Apparel: Clothing brands, footwear, and accessories."
    ],
    "Pharmaceuticals": [
        "Prescription Medication: Antibiotics, antivirals, and vaccines.",
        "Over-The-Counter: Pain relievers, cold medication, and vitamins.",
        "Medical Supplies: Bandages, syringes, and medical equipment."
    ],
    "Travel and Tourism": [
        "Hotels and Resorts: Luxury hotels, resorts, and budget stays.",
        "Airlines and Cruises: Flight services, cruises, and related services.",
        "Tourist Attractions: Museums, parks, and sightseeing."
    ],
    "Real Estate": [
        "Residential: Apartments, houses, and housing projects.",
        "Commercial: Office spaces, shopping malls, and warehouses.",
        "Property Management: Real estate agencies, and property listing services."
    ],
    "Education": [
        "K-12: Kindergartens, elementary schools, and high schools.",
        "Higher Education: Colleges, universities, and vocational schools.",
        "Online Learning: E-courses, webinars, and online certifications."
    ],
    "Non-profits and Social Causes": [
        "Charities: Fundraising, donations, and charitable events.",
        "Activism: Environmental, social justice, and other campaigns.",
        "Community Services: Food banks, shelters, and local services."
    ],
    "Other": [
        "Miscellaneous: Any other subcategory not listed above."
    ]
}

LANGUAGES = {
    1: "Bahasa Indonesia",
    2: "English",
    3: "Bahasa Daerah",
    4: "Others",
    5: "None"
}

REGIONS = {
    1: "Indonesia",
    2: "South East Asia",
    3: "Worldwide"
}

VOICE_TYPES = {
    1: "Voice Over",
    2: "Vocal",
    3: "Both"
}

AGE_DEMOGRAPHICS = {
    1: 'Babies (0-4)',
    2: 'Kids (5-9)',
    3: 'Teens (10-19)',
    4: 'Adults (20-39)',
    5: 'Middle-Aged (40-64)',
    6: 'Seniors (65+)'
}

CHARACTERS = {
    1: "High",
    2: "Mid",
    3: "Deep/Low"
}

GENDERS = {
    1: "Male",
    2: "Female"
}

def transcribe_audio(audio_file, duration_for_noise=0.5):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        recognizer.adjust_for_ambient_noise(source, duration=duration_for_noise)
        
        # Segment the audio (optional)
        # audio_segments = [recognizer.record(source, duration=10) for _ in range(3)]  # 10-second segments
        
        audio_data = recognizer.record(source)
        
        # Multiple recognition attempts (optional)
        # attempts = []
        try:
            text = recognizer.recognize_google(audio_data, language='id-ID')
            
            # attempts.append(text)  # For multiple attempts
            
            return text  # or max(set(attempts), key=attempts.count) for multiple attempts
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")
            return None
        
def extract_named_entities(sentence):
    named_entities = []
    for chunk in ne_chunk(pos_tag(word_tokenize(sentence))):
        if type(chunk) == Tree:
            entity_name = " ".join([token for token, pos in chunk.leaves()])
            named_entities.append(entity_name.lower())
    return named_entities

def summarize_text_advanced(text, num_sentences=3):
    # Step 1: Split text into sentences
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s) > 0]
    
    # Step 2: Tokenize and Score
    word_freq = Counter()
    named_entities = Counter()
    common_nouns = Counter()
    
    for sentence in sentences:
        for word, pos in pos_tag(word_tokenize(sentence.lower())):
            if word not in stop_words:
                word_freq[word] += 1
                
            # Count common nouns for thematic clustering
            if pos == 'NN':
                common_nouns[word] += 1
        
        for entity in extract_named_entities(sentence):
            named_entities[entity] += 1

    print("Named Entities Count:", named_entities)
    print("Common Nouns Count:", common_nouns)
    
    sentence_scores = {}
    for sentence in sentences:
        # Basic sentence score based on word frequency
        sentence_scores[sentence] = sum([word_freq[word.lower()] for word in word_tokenize(sentence.lower()) if word.lower() not in stop_words])
        
        # Adding additional weight for named entities
        for entity in extract_named_entities(sentence):
            sentence_scores[sentence] += named_entities[entity]
            
        # Adding additional weight for common nouns (Thematic Clustering)
        for word, pos in pos_tag(word_tokenize(sentence.lower())):
            if pos == 'NN':
                sentence_scores[sentence] += common_nouns[word]
        
        # Adding additional weight for sentiment (Sentiment Analysis)
        sentiment_score = sia.polarity_scores(sentence)['compound']
        sentence_scores[sentence] += abs(sentiment_score) * 10  # Multiply by 10 to give it more weight
        print(f"Sentiment for '{sentence}': {sentiment_score}")
    
    # Step 3: Select Top Sentences
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    summary_sentences = [s[0] for s in sorted_sentences[:num_sentences]]
    
    # Return the summary along with additional metrics
    return {
        'named_entities': named_entities,
        'common_nouns': common_nouns,
        'sentiment_score': sentiment_score,
    }

def summarize_text(text, num_sentences=3):
    # Step 1: Split text into sentences
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if len(s) > 0]
    
    # Step 2: Tokenize and Score
    word_freq = Counter()
    for sentence in sentences:
        for word in sentence.lower().split():
            if word not in stop_words:
                word_freq[word] += 1

    sentence_scores = {}
    for sentence in sentences:
        sentence_scores[sentence] = sum([word_freq[word.lower()] for word in sentence.split() if word.lower() not in stop_words])
        
    # Step 3: Select Top Sentences
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    summary_sentences = [s[0] for s in sorted_sentences[:num_sentences]]
    return ' '.join(summary_sentences)

def categorize_energy(y, sr):
    # Energy approximation
    energy_value = categorize_energy(y, sr)

    # Categorize the continuous energy_value into one of your existing categories
    if energy_value > 0.7:  # You'll need to determine these thresholds empirically
        energy = "lively and energetic"
    elif energy_value > 0.3:
        energy = "moderate"
    else:
        energy = "soft and subdued"

import tensorflow as tf
import joblib
# Load the trained model, scaler, and label encoder
model = tf.keras.models.load_model('C:\\Users\\USER\\Documents\\Agson Sonics Python Codes\\best_model.h5', compile=False)
scaler = joblib.load('C:\\Users\\USER\\Documents\\Agson Sonics Python Codes\\scaler.pkl')
label_encoder = joblib.load('C:\\Users\\USER\\Documents\\Agson Sonics Python Codes\\label_encoder.pkl')

def separate_and_measure_using_spleeter(file_path):
    if not os.path.exists(file_path):
        print("Input file does not exist!")
        return None, None
    
    # Use spleeter to separate audio
    output_dir = "spleeter_output"
    base_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
    output_subdir = os.path.join(output_dir, base_name_without_ext)
    
    cmd = f'"spleeter" separate "{file_path}" -o "{output_dir}" -p spleeter:2stems'
    subprocess.run(cmd, shell=True)
    
    # Load separated tracks
    vocals_path = os.path.join(output_subdir, "vocals.wav")
    accompaniment_path = os.path.join(output_subdir, "accompaniment.wav")

    if not os.path.exists(vocals_path) or not os.path.exists(accompaniment_path):
        print(f"Expected output files not found for {file_path}!")
        return None, None
    
    # Predict genre for accompaniment
    preprocessed_audio = preprocess_audio(accompaniment_path, scaler)
    if preprocessed_audio is not None:
        predictions = model.predict(preprocessed_audio)
        predicted_genre_idx = np.argmax(predictions)
        predicted_genre = label_encoder.inverse_transform([predicted_genre_idx])[0]
        print(f"Accompaniment File: {accompaniment_path}, Predicted Genre: {predicted_genre}")

    y_vocals, sr = librosa.load(vocals_path, sr=None)
    y_accompaniment, sr = librosa.load(accompaniment_path, sr=None)

    # Measure RMS of each track
    rms_vocals = librosa.feature.rms(y=y_vocals)[0]
    rms_accompaniment = librosa.feature.rms(y=y_accompaniment)[0]

    # Convert RMS to dB
    avg_db_vocals = rms_to_db(np.mean(rms_vocals))
    avg_db_accompaniment = rms_to_db(np.mean(rms_accompaniment))

    # Measure max dB of each track
    max_db_vocals = np.max(librosa.amplitude_to_db(np.abs(librosa.stft(y_vocals)), ref=np.max))
    max_db_accompaniment = np.max(librosa.amplitude_to_db(np.abs(librosa.stft(y_accompaniment)), ref=np.max))

    return max_db_vocals, max_db_accompaniment, avg_db_vocals, avg_db_accompaniment

def preprocess_audio(audio_file_path, scaler):
    try:
        audio, sr = librosa.load(audio_file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr)
        
        # Adjust MFCC to match the expected number of time frames
        if mfcc.shape[1] < 430:
            padding_width = 430 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, padding_width)), mode='constant')
        else:
            mfcc = mfcc[:, :430]
        
        mfcc = scaler.transform(mfcc.flatten().reshape(1, -1)).flatten()
        mfcc = mfcc.reshape(1, 20, 430, 1)
        
        return mfcc
    except Exception as e:
        print(f"Could not process audio file {audio_file_path}: {e}")
        return None

def process_audio_file_updated(file_path):
    print ("starting process_audio_file_updated")

    # Construct the path for the accompaniment track based on the provided file_path
    accompaniment_path = os.path.join("spleeter_output", os.path.splitext(os.path.basename(file_path))[0], "accompaniment.wav")
    
    # Check if the accompaniment file exists
    if not os.path.exists(accompaniment_path):
        print(f"Accompaniment file {accompaniment_path} does not exist! Have you run separate_and_measure_using_spleeter() yet?")
        return
    
    # Load only the accompaniment track for further analysis
    y, sr = librosa.load(accompaniment_path, sr=None)

    # Normalize the audio to 0 dBFS
    def normalize_audio(audio):
        return audio / np.max(np.abs(audio))

    y = normalize_audio(y)  # Normalize the audio before any further processing

    # Calculate the chromagram of the audio signal
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

    # Sum the chromagram to get the energy for each pitch class
    chroma_sum = np.sum(chromagram, axis=1)

    # Find the key (it will be an integer index)
    key = np.argmax(chroma_sum)

    # Translate that to a key in Spotify's representation
    key_spotify = key

    # Separate harmonic (voice) and percussive (music) components
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    
    # Compute RMS for voice and music
    rms_voice = librosa.feature.rms(y=y_harmonic)[0]
    rms_music = librosa.feature.rms(y=y_percussive)[0]
    
    # Convert RMS to dB
    def rms_to_db(rms_val):
        return 20 * np.log10(rms_val)
    
    avg_db_voice = rms_to_db(np.mean(rms_voice))
    avg_db_music = rms_to_db(np.mean(rms_music))
    
    # Calculate MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.round(mfccs, 4)
    stats = []
    for i in range(mfccs.shape[0]):
        stats.extend([np.mean(mfccs[i, :]), np.std(mfccs[i, :]), np.min(mfccs[i, :]), np.max(mfccs[i, :]), np.median(mfccs[i, :])])
    sonic_identifier = hashlib.sha256(''.join(map(str, stats)).encode()).hexdigest()
    file_name = os.path.basename(file_path)
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    pitches, magnitudes = librosa.piptrack(y=y_harmonic, sr=sr)
    avg_pitch_val = np.mean(pitches[pitches > 0])
    if avg_pitch_val < 20:
        avg_pitch = "Infrasound"
    elif avg_pitch_val >= 20 and avg_pitch_val < 60:
        avg_pitch = "Sub-bass"
    elif avg_pitch_val >= 60 and avg_pitch_val < 250:
        avg_pitch = "Bass"
    elif avg_pitch_val >= 250 and avg_pitch_val < 500:
        avg_pitch = "Low Midrange"
    elif avg_pitch_val >= 500 and avg_pitch_val < 2000:
        avg_pitch = "Midrange"
    elif avg_pitch_val >= 2000 and avg_pitch_val < 4000:
        avg_pitch = "Upper Midrange"
    elif avg_pitch_val >= 4000 and avg_pitch_val < 6000:
        avg_pitch = "Presence"
    elif avg_pitch_val >= 6000 and avg_pitch_val < 20000:
        avg_pitch = "Brilliance"
    else:
        avg_pitch = "Ultrasonic"
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    if tempo > 160:
        tempo_label = "Very Fast"
    elif tempo > 120:
        tempo_label = "Fast"
    elif tempo > 100:
        tempo_label = "Medium"
    elif tempo > 60:
        tempo_label = "Moderate Slow"
    elif tempo > 40:
        tempo_label = "Slow"
    else:
        tempo_label = "Very Slow"

    def normalize(x, min_val, max_val):
        return (x - min_val) / (max_val - min_val)

    def calculate_energy_dynamic_range(y):
    # Calculate dynamic range
        dynamic_range = np.max(y) - np.min(y)
        
        # Normalize the dynamic range to fit within [0, 1]
        normalized_dynamic_range = dynamic_range / 2.0
        
        return normalized_dynamic_range

    # Calculating energy using dynamic range
    energy_dynamic_range = calculate_energy_dynamic_range(y)

    # Acousticness approximation: Revised formula
    acousticness = np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-10)

    # Calculate RMS loudness in dB
    rms_value = np.mean(librosa.feature.rms(y=y)[0])
    loudness = 20 * np.log10(rms_value)
    
    # Convert acousticness and loudness to categorical variables
    acousticness_label = "high" if acousticness > 0.7 else "medium" if acousticness > 0.3 else "low"

    # Convert loudness to a categorical variable
    loudness_label = "high" if loudness > -20 else "medium" if loudness > -40 else "low"

    # Danceability approximation: Revised formula
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    danceability = np.mean(onset_env) / 100  # Normalized

    # Valence approximation: Existing + fine-tuning
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    valence = np.mean(tonnetz)

    # Add tempo factor
    tempo_factor = 0.0
    if tempo_label == "Fast" or tempo_label == "Very Fast":
        tempo_factor = 0.2  # Fine-tune this value
    valence += tempo_factor

    # Normalize valence to be within [0, 1]
    valence = min(max(valence, 0.0), 1.0)

    # Energy approximation: Revised formula
    energy_value = np.mean(librosa.feature.rms(y=y))

    # Make sure energy_value is between 0.0 and 1.0
    energy_value = min(max(energy_value, 0.0), 1.0)

    # Step 1: Call fuzzy_genre_classification to get the genre percentages
    genre_percentages = fuzzy_genre_classification(
    energy_value, tempo_label, acousticness_label, loudness_label, danceability, valence
)
    print("Predicted genre percentages:", genre_percentages)
    
    def normalize_to_range(value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)

    # Calculate danceability
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    danceability = normalize_to_range(np.mean(onset_env), np.min(onset_env), np.max(onset_env))

    # 1. Statistical Features
    stat_features = []
    stat_features.append(np.var(y_harmonic))
    stat_features.append(skew(y_harmonic))
    stat_features.append(kurtosis(y_harmonic))

    # 2. Spectral Features
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # 3. MFCCs
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    # 4. Wavelets
    coeffs = pywt.wavedec(y, 'db1', level=4)
    cA4, cD4, cD3, cD2, cD1 = coeffs
    wavelet_features = [np.mean(cA4), np.mean(cD4)]

    # 5. Network Metrics (Centrality)
    # Create a simple example graph based on MFCCs (this is a very rudimentary example)
    G = nx.Graph()
    for i in range(len(mfccs)):
        for j in range(i+1, len(mfccs)):
            G.add_edge(i, j, weight=abs(mfccs[i] - mfccs[j]))
        centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
        network_features = [v for k, v in centrality.items()]

    # Combine features
    all_features = np.concatenate([stat_features, [spectral_contrast, spectral_bandwidth], mfccs, wavelet_features, network_features])

    # Instrumentalness: Revised formula
    instrumentalness = np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-10)

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    brightness = "bright" if np.mean(spec_centroid) > 2000 else "dark"
    rhythm = "steady" if np.std(np.diff(mfccs)) < 0.5 else "variable"
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    sound_dynamics = "variable" if np.mean(spec_contrast) > 20 else "steady"
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    scale = "major" if np.mean(tonnetz) > 0 else "minor"

    # Add musical scale factor
    scale_factor = 0 if scale == "minor" else 0.1  # Fine-tune this value
    valence += scale_factor
    
    
    insights = {
        "Average Pitch": avg_pitch,
        "Energy": energy_value,
        "Sound Dynamics": sound_dynamics,
        "Tempo": tempo_label,
        "Average Brightness": brightness,
        "Musical Scale": scale,
        "Rhythmic Stability": rhythm,
        "Avg_Pitch_Val": avg_pitch_val,
        "Tempo_Val": tempo,
        "Spec_Centroid_Mean": np.mean(spec_centroid),
        "max_db_voice": np.max(librosa.amplitude_to_db(np.abs(librosa.stft(y_harmonic)), ref=np.max)),
        "max_db_music": np.max(librosa.amplitude_to_db(np.abs(librosa.stft(y_percussive)), ref=np.max)),
        "avg_db_voice": avg_db_voice,
        "avg_db_music": avg_db_music,
        "acousticness": acousticness,
        "danceability": danceability,
        "energy": energy_dynamic_range,
        "instrumentalness": instrumentalness,
        "loudness": loudness,
        "valence": valence,
        "key": key_spotify,
    }
    
    # Now you can call your function to align these insights with Spotify parameters
    spotify_params = align_insights_to_spotify(insights)

    # And get Spotify recommendations
    access_token = get_spotify_access_token()  # Assuming you have this function defined
    recommendations = get_spotify_recommendations(access_token, spotify_params, genre_percentages)  # Pass genre_percentages here

    # Print the recommendations (or handle them however you like)
    print("Spotify Recommendations:", recommendations)
    print("done process_audio_file_updated")
    return {'File Name': file_name, 'Sonic Identifier': sonic_identifier, **insights}

def fuzzy_genre_classification(energy, tempo_label, acousticness, loudness, danceability, valence):
    print("Extended Fuzzy Genre Classification is starting...")
    
    genre_percentages = {
        "acoustic": 0, "afrobeat": 0, "alt-rock": 0, "alternative": 0, "ambient": 0, "anime": 0, "ballads": 0,
        "black-metal": 0, "bluegrass": 0, "blues": 0, "bossanova": 0, "brazil": 0, "breakbeat": 0,
        "british": 0, "cantopop": 0, "chicago-house": 0, "children": 0, "chill": 0, "classical": 0,
        "club": 0, "comedy": 0, "country": 0, "dance": 0, "dancehall": 0, "death-metal": 0,
        "deep-house": 0, "detroit-techno": 0, "disco": 0, "disney": 0, "drum-and-bass": 0, "dub": 0,
        "dubstep": 0, "edm": 0, "electro": 0, "electronic": 0, "emo": 0, "folk": 0, "forro": 0,
        "french": 0, "funk": 0, "garage": 0, "german": 0, "gospel": 0, "goth": 0, "grindcore": 0,
        "groove": 0, "grunge": 0, "guitar": 0, "happy": 0, "hard-rock": 0, "hardcore": 0, "hardstyle": 0,
        "heavy-metal": 0, "hip-hop": 0, "holidays": 0, "honky-tonk": 0, "house": 0, "idm": 0,
        "indian": 0, "indie": 0, "indie-pop": 0, "industrial": 0, "iranian": 0, "j-dance": 0,
        "j-idol": 0, "j-pop": 0, "j-rock": 0, "jazz": 0, "k-pop": 0, "kids": 0, "latin": 0,
        "latino": 0, "malay": 0, "mandopop": 0, "metal": 0, "metal-misc": 0, "metalcore": 0,
        "minimal-techno": 0, "movies": 0, "mpb": 0, "new-age": 0, "new-release": 0, "opera": 0,
        "pagode": 0, "party": 0, "philippines-opm": 0, "piano": 0, "pop": 0, "pop-film": 0,
        "post-dubstep": 0, "power-pop": 0, "progressive-house": 0, "psych-rock": 0, "punk": 0,
        "punk-rock": 0, "r-n-b": 0, "rainy-day": 0, "reggae": 0, "reggaeton": 0, "road-trip": 0,
        "rock": 0, "rock-n-roll": 0, "rockabilly": 0, "romance": 0, "sad": 0, "salsa": 0, "samba": 0,
        "sertanejo": 0, "show-tunes": 0, "singer-songwriter": 0, "ska": 0, "sleep": 0, "songwriter": 0,
        "soul": 0, "soundtracks": 0, "spanish": 0, "study": 0, "summer": 0, "swedish": 0,
        "synth-pop": 0, "tango": 0, "techno": 0, "trance": 0, "trip-hop": 0, "turkish": 0,
        "work-out": 0, "world-music": 0
    }

    # Define feature importance weights
    weights = {
        'energy': 1.5,
        'tempo': 1.2,
        'acousticness': 1,
        'loudness': 1.3,
        'danceability': 1,
        'valence': 1
    }
    
    energy_mapping = {
        "lively and energetic": {
            'rock': 30, 'pop': 30, 'hip-hop': 30, 'jazz': 10,
            'dance': 25, 'edm': 25, 'punk': 30, 'metal': 35,
            'reggae': 20, 'ska': 25, 'latin': 20, 'funk': 20,
            'r-n-b': 20, 'house': 25, 'techno': 25, 'dubstep': 30,
            'drum-and-bass': 30, 'trance': 25, 'hardcore': 35, 'grunge': 30,
            'afrobeat': 25, 'reggaeton': 25, 'samba': 20, 'salsa': 20
        },
        "moderate": {
            'jazz': 30, 'electronic': 30, 'country': 20, 'blues': 20,
            'rock': 20, 'folk': 20, 'indie': 20, 'gospel': 15,
            'soul': 15, 'swing': 15, 'world-music': 15, 'alternative': 20,
            'industrial': 20, 'psychedelic': 20, 'new-age': 20, 'ambient': 20,
            'bluegrass': 20, 'americana': 20, 'soundtracks': 15, 'bossanova': 15,
            'tango': 15, 'flamenco': 15, 'trip-hop': 20, 'breakbeat': 20,
            'dub': 20, 'garage': 20, 'grindcore': 20, 'emo': 20
        },
        "soft and subdued": {
            'classical': 50, 'blues': 30, 'jazz': 20, 'rock': 10,
            'acoustic': 40, 'ballads': 35, 'chill': 35, 'lounge': 35,
            'sleep': 40, 'easy-listening': 40, 'opera': 35, 'orchestral': 35,
            'children': 15, 'comedy': 10, 'spoken-word': 10, 'poetry': 10,
            'holiday': 15, 'religious': 15, 'worship': 15, 'celtic': 20,
            'native-american': 20, 'environmental': 20, 'healing': 20,
            'meditative': 40, 'nature': 40, 'calm': 40, 'relax': 40
        }
    }
 
    tempo_mapping = {
        "Very Fast": {
            'hip-hop': 30, 'rock': 20, 'punk': 25, 'metal': 20, 
            'reggae': 20, 'techno': 25, 'ska': 20, 'jazz': 15,
            'drum-and-bass': 25, 'trance': 20, 'hardcore': 25, 'grindcore': 25,
            'salsa': 20, 'afrobeat': 25
        },
        "Fast": {
            'rock': 30, 'electronic': 20, 'punk-rock': 25, 'r-n-b': 15,
            'dance': 20, 'house': 20, 'funk': 20, 'latin': 20,
            'reggaeton': 20, 'samba': 20, 'breakbeat': 20, 'dubstep': 20,
            'garage': 20, 'emo': 15
        },
        "Medium": {
            'pop': 30, 'jazz': 30, 'country': 25, 'blues': 20,
            'folk': 25, 'indie': 25, 'gospel': 20, 'soul': 20,
            'swing': 20, 'alternative': 20, 'bluegrass': 20, 
            'americana': 20, 'bossanova': 20, 'tango': 20,
            'flamenco': 20, 'trip-hop': 20
        },
        "Moderate Slow": {
            'country': 30, 'blues': 30, 'classical': 20, 'acoustic': 20,
            'ballads': 20, 'chill': 20, 'lounge': 20, 'opera': 20,
            'orchestral': 20, 'ambient': 20, 'new-age': 20,
            'soundtracks': 20, 'easy-listening': 20
        },
        "Slow": {
            'classical': 50, 'blues': 20, 'gospel': 20, 'jazz': 20,
            'soul': 20, 'worship': 20, 'religious': 20,
            'celtic': 20, 'native-american': 20, 'environmental': 20,
            'healing': 20, 'meditative': 20, 'nature': 20
        },
        "Very Slow": {
            'classical': 50, 'opera': 25, 'ambient': 25, 'new-age': 25,
            'soundtracks': 25, 'spoken-word': 25, 'poetry': 25,
            'healing': 25, 'meditative': 25
        }
    }
    
    acousticness_mapping = {
        "high": {
            'classical': 30, 'jazz': 30, 'blues': 25, 'folk': 30,
            'country': 20, 'acoustic': 35, 'ambient': 20, 'gospel': 25,
            'soul': 20, 'reggae': 15, 'bossanova': 30, 'world-music': 25,
            'new-age': 20, 'soundtracks': 20, 'opera': 35
        },
        "medium": {
            'pop': 30, 'country': 20, 'r-n-b': 25, 'indie': 20,
            'rock': 15, 'hip-hop': 15, 'electronic': 15, 'latin': 20,
            'reggaeton': 20, 'dance': 20, 'singer-songwriter': 25,
            'ballads': 20, 'chill': 20, 'lounge': 20, 'alternative': 20
        },
        "low": {
            'rock': 30, 'hip-hop': 35, 'electronic': 35, 'metal': 30,
            'punk': 25, 'techno': 35, 'house': 30, 'dubstep': 35,
            'trance': 30, 'edm': 35, 'industrial': 30, 'grunge': 25,
            'hardcore': 25, 'disco': 25, 'funk': 25, 'breakbeat': 30,
            'club': 30, 'dub': 25
        }
    }

    loudness_mapping = {
        "high": {
            'rock': 30, 'hip-hop': 30, 'electronic': 30, 'metal': 35,
            'punk': 25, 'techno': 35, 'house': 30, 'dubstep': 35,
            'trance': 30, 'edm': 35, 'industrial': 30, 'grunge': 25,
            'hardcore': 25, 'reggaeton': 25, 'dancehall': 25, 'dance': 25,
            'breakbeat': 30, 'club': 30, 'dub': 25, 'drum-and-bass': 30
        },
        "medium": {
            'pop': 30, 'country': 25, 'r-n-b': 25, 'jazz': 20,
            'blues': 20, 'indie': 20, 'alternative': 20, 'reggae': 20,
            'soul': 20, 'funk': 20, 'disco': 20, 'singer-songwriter': 20,
            'latin': 25, 'ballads': 20, 'acoustic': 20, 'chill': 20,
            'lounge': 20, 'world-music': 20, 'gospel': 20, 'folk': 20,
            'afrobeat': 20, 'forro': 15, 'salsa': 20, 'samba': 20,
            'sertanejo': 20, 'cantopop': 20, 'mandopop': 20
        },
        "low": {
            'classical': 40, 'opera': 35, 'ambient': 35, 'new-age': 35,
            'soundtracks': 35, 'bossa nova': 30, 'chillout': 30,
            'easy listening': 30, 'instrumental': 30, 'children': 30,
            'comedy': 25, 'spoken word': 30, 'study': 30, 'sleep': 40,
            'relax': 35, 'meditation': 35, 'yoga': 30, 'nature sounds': 40
        }
    }


    danceability_mapping = {
        "high": {
            'pop': 30, 'electronic': 35, 'hip-hop': 40, 'r-n-b': 35,
            'reggaeton': 35, 'dancehall': 35, 'dance': 40, 'techno': 30,
            'house': 35, 'dubstep': 30, 'trance': 35, 'reggae': 25,
            'soul': 30, 'funk': 30, 'disco': 40, 'afrobeat': 25,
            'latin': 25, 'salsa': 25, 'samba': 25, 'sertanejo': 25,
            'forro': 25, 'cantopop': 25, 'mandopop': 25, 'k-pop': 30,
            'j-pop': 30, 'club': 35, 'breakbeat': 30, 'dub': 25
        },
        "medium": {
            'rock': 25, 'country': 20, 'jazz': 25, 'blues': 20,
            'indie': 20, 'alternative': 20, 'ballads': 20, 'singer-songwriter': 20,
            'acoustic': 20, 'world-music': 20, 'gospel': 20, 'folk': 20,
            'punk': 20, 'industrial': 20, 'grunge': 20, 'hardcore': 20,
            'metal': 20, 'drum-and-bass': 25, 'emo': 20, 'goth': 20
        },
        "low": {
            'classical': 15, 'opera': 15, 'ambient': 15, 'new-age': 15,
            'soundtracks': 15, 'bossa nova': 15, 'chillout': 15,
            'easy listening': 15, 'instrumental': 15, 'children': 15,
            'comedy': 15, 'spoken word': 15, 'study': 15, 'sleep': 15,
            'relax': 15, 'meditation': 15, 'yoga': 15, 'nature sounds': 15
        }
    }


    valence_mapping = {
        "high": {
            'pop': 30, 'country': 30, 'salsa': 30, 'reggaeton': 30,
            'dancehall': 30, 'samba': 30, 'sertanejo': 30, 'forro': 30,
            'latin': 30, 'disco': 30, 'funk': 30, 'reggae': 25,
            'soul': 30, 'afrobeat': 25, 'k-pop': 25, 'j-pop': 25,
            'house': 25, 'techno': 25, 'dubstep': 25, 'trance': 25,
            'cantopop': 25, 'mandopop': 25, 'club': 25, 'dance': 30,
            'breakbeat': 25, 'dub': 25, 'happy': 40, 'summer': 35
        },
        "medium": {
            'rock': 25, 'blues': 25, 'jazz': 25, 'alternative': 25,
            'indie': 25, 'punk': 25, 'grunge': 25, 'emo': 25,
            'goth': 25, 'metal': 20, 'world-music': 25, 'folk': 25,
            'gospel': 25, 'ballads': 25, 'singer-songwriter': 25,
            'industrial': 25, 'hip-hop': 20, 'r-n-b': 20, 'hardcore': 20,
            'drum-and-bass': 20, 'electronic': 20, 'electro': 20,
            'idm': 20, 'deep-house': 20, 'detroit-techno': 20,
            'minimal-techno': 20, 'progressive-house': 20, 'groove': 20
        },
        "low": {
            'classical': 20, 'opera': 20, 'ambient': 20, 'new-age': 20,
            'soundtracks': 20, 'bossa nova': 20, 'chillout': 20,
            'acoustic': 20, 'easy listening': 20, 'instrumental': 20,
            'children': 20, 'comedy': 20, 'spoken word': 20, 'study': 20,
            'sleep': 20, 'relax': 20, 'meditation': 20, 'yoga': 20,
            'nature sounds': 20, 'movies': 20, 'anime': 20,
            'disney': 20, 'show-tunes': 20, 'sad': 20, 'rainy-day': 20
        }
    }

    # Feature Weights
    feature_weights = {'energy': 0.2, 'tempo': 0.2, 'acousticness': 0.2, 'loudness': 0.1, 'danceability': 0.1, 'valence': 0.2}

    def update_percentages(mapping, label, weight=1):
        for genre, value in mapping.get(label, {}).items():
            genre_percentages[genre] += value * weight

    # Update genre_percentages based on the provided features
    for genre, value in energy_mapping.get(energy, {}).items():
        genre_percentages[genre] += value * weights['energy']

    for genre, value in tempo_mapping.get(tempo_label, {}).items():
        genre_percentages[genre] += value * weights['tempo']

    for genre, value in acousticness_mapping.get(acousticness, {}).items():
        genre_percentages[genre] += value * weights['acousticness']

    for genre, value in loudness_mapping.get(loudness, {}).items():
        genre_percentages[genre] += value * weights['loudness']

    for genre, value in danceability_mapping.get(danceability, {}).items():
        genre_percentages[genre] += value * weights['danceability']

    for genre, value in valence_mapping.get(valence, {}).items():
        genre_percentages[genre] += value * weights['valence']
    
    # Special conditions for Jazz
    if tempo_label == "Medium" and energy < 0.5:
        genre_percentages['Jazz'] += 20  # domain knowledge
    if acousticness == "high" and danceability == "low":
        genre_percentages['Jazz'] += 15  # domain knowledge

    # Special conditions for Rock
    if tempo_label == "Fast" and energy > 0.7:
        genre_percentages['Rock'] += 20  # domain knowledge
    
    # Special conditions for Classical
    if tempo_label == "Slow" and acousticness == "high":
        genre_percentages['Classical'] += 20  # domain knowledge
    
    # Special conditions for Electronic
    if danceability == "high" and energy > 0.7:
        genre_percentages['Electronic'] += 15  # domain knowledge
    
    # Special conditions for Hip-Hop
    if tempo_label == "Very Fast" and danceability == "high":
        genre_percentages['Hip-Hop'] += 20  # domain knowledge
    
    # Normalize the percentages to sum to 100
    total = sum(genre_percentages.values())
    for genre in genre_percentages:
        genre_percentages[genre] = (genre_percentages[genre] / total) * 100 if total != 0 else 0
    
    return genre_percentages

# Call the function and print the result
result = fuzzy_genre_classification("lively and energetic", "Very Fast", "low", "high", "medium", "medium")

#Spotify initialization
print("Initializing Spotify...")
# Your client ID and client secret
client_id = "506b531b647148b5a7ee0643a7ab9658"
client_secret = "a1de531300bc486481ced3c2be6e6ce7"
redirect_uri = "https://localhost:8080/"

# Specify cache path
cache_path = "C:\\Users\\USER\\Documents\\Agson Sonics Python Codes\\.cache"
print("Spotify initialized!")

def align_insights_to_spotify(insights):
    print("Aligning insights to Spotify...")
    spotify_params = {}
    
    # Validate and map insights to Spotify's API
    def clip(value, lower, upper, round_to_tenth=False):
        value = max(min(value, upper), lower)
        if round_to_tenth:
            return round(round(value * 10) / 10, 1)  # Round to nearest 0.1
        else:
            return int(round(value))  # Round to nearest integer
    
    # Keep the decimal places by removing the round function
    spotify_params['target_acousticness'] = insights['acousticness']
    spotify_params['target_danceability'] = insights['danceability']
    spotify_params['target_energy'] = insights['energy']
    spotify_params['target_instrumentalness'] = insights['instrumentalness']
    spotify_params['target_valence'] = insights['valence']
    
    
    # Round to the nearest integer for tempo and loudness
    spotify_params["target_tempo"] = clip(insights.get("Tempo_Val", 0), 0, 500)
    spotify_params["target_loudness"] = clip(insights.get("loudness", -60), -60, 0)
    spotify_params["target_key"] = clip(insights.get("key", 0), 0, 11)

    # Map the calculated musical scale to Spotify's mode attribute
    spotify_params['mode'] = 1 if insights['Musical Scale'] == 'major' else 0
    
    print("Validated and Selectively Rounded Spotify Parameters:", spotify_params)
    
    return spotify_params

def get_spotify_recommendations(access_token, spotify_params, genre_percentages):
    print("Getting Spotify recommendations...")
    
    # Create a mapping between fuzzy logic genres and Spotify-acceptable genres
    genre_mapping = {
        'rock': ['rock', 'alt-rock', 'hard-rock', 'punk-rock', 'rock-n-roll', 'grunge', 'industrial'],
        'jazz': ['jazz', 'bossanova'],
        'classical': ['classical', 'opera', 'instrumental'],
        'pop': ['pop', 'indie-pop', 'power-pop', 'pop-film', 'synth-pop', 'k-pop', 'j-pop'],
        'hip-hop': ['hip-hop', 'r-n-b'],
        'electronic': ['electronic', 'deep-house', 'detroit-techno', 'dubstep', 'edm', 'electro', 'house', 'techno', 'trance', 'idm', 'ambient'],
        'country': ['country', 'bluegrass', 'honky-tonk', 'country-pop'],
        'blues': ['blues'],
        'metal': ['metal', 'heavy-metal', 'death-metal', 'metalcore', 'metal-misc'],
        'folk': ['folk', 'acoustic', 'singer-songwriter'],
        'reggae': ['reggae', 'dancehall', 'ska'],
        'soul': ['soul', 'gospel', 'funk'],
        'latin': ['latin', 'reggaeton', 'salsa', 'samba', 'latino'],
        'dance': ['dance', 'club', 'breakbeat', 'drum-and-bass'],
        'world-music': ['world-music', 'afrobeat', 'forro'],
        'alternative': ['alternative', 'alt-rock', 'emo', 'goth'],
        'soundtracks': ['soundtracks', 'movies', 'disney', 'anime', 'show-tunes'],
        'comedy': ['comedy'],
        'children': ['children', 'kids'],
        'easy-listening': ['easy-listening', 'chill', 'lounge'],
        'new-age': ['new-age', 'ambient', 'meditation', 'yoga', 'nature-sounds', 'relax']
    }

    # Calculate the main genre with the highest percentage
    transformed_genres = {genre_mapping[k][0]: v for k, v in genre_percentages.items() if k in genre_mapping}
    sorted_genres = sorted(transformed_genres.items(), key=lambda x: x[1], reverse=True)
    main_genre = sorted_genres[0][0] if sorted_genres else None  # take the first one
    
    # Find corresponding sub-genres
    sub_genres = genre_mapping.get(main_genre, [])
    
    # Print the main and sub-genres
    print(f"The main genre is {main_genre} and the sub-genres are {sub_genres}")
    
    # Use main genre and sub-genres as seed genres
    seed_genres = [main_genre] + sub_genres
    
    # Join them into a comma-separated string
    seed_genres_str = ",".join(seed_genres)
    print(f"Seed genres used for recommendations: {seed_genres_str}")
    
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    params = {
        "limit": 5,
        "seed_genres": seed_genres
    }
    
    params.update(spotify_params)
    
    url = "https://api.spotify.com/v1/recommendations"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            recommendations = response.json()
            track_urls = [track['external_urls']['spotify'] for track in recommendations['tracks']]
            for i, track_url in enumerate(track_urls):
                print(f"Recommendation {i+1}: {track_url}")
            return track_urls
        else:
            return {"error": f"Failed to get recommendations, status code: {response.status_code}, message: {response.text}"}
    except Exception as e:
        print(f"An exception occurred: {e}")
        return {"error": "An exception occurred while fetching recommendations"}   

def get_spotify_access_token():
    # Check if token is already cached
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            token_info = json.load(f)
            if time.time() < token_info["expires_at"]:
                return token_info["access_token"]

    # If not cached, fetch new token
    auth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope="user-read-recently-played user-top-read playlist-modify-public",
        cache_path=cache_path
    )
    auth_url = auth.get_authorize_url()
    print(f"Please go to the following URL to authenticate: {auth_url}")
    response_code = input("Enter the response code: ")
    auth.get_access_token(response_code)
    return auth.get_access_token(response_code, as_dict=False)

def test_spotify_token(access_token):
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get("https://api.spotify.com/v1/me", headers=headers)
    if response.status_code == 200:
        print("Token is valid. User information fetched successfully.")
        print(response.json())
    else:
        print(f"Token might be invalid. Status code: {response.status_code}, Message: {response.text}")

if __name__ == "__main__":
    access_token = get_spotify_access_token()
    test_spotify_token(access_token)

def rms_to_db(rms_val):
    return 20 * np.log10(rms_val)

def play_audio_and_get_label(file_path):
    insights = {}
    multiple_VO = False
    multiple_Vocal = False
    
    while True:  # Start of the loop to re-label if necessary

        cmd = ["C:\\Program Files (x86)\\Windows Media Player\\wmplayer.exe", file_path]
        subprocess.run(cmd)
        file_name = os.path.basename(file_path)
        main_label = input(f"Please label the audio file '{file_name}': ")
        brand_name = input("Brand? ")
        for key, (industry, description) in INDUSTRY_OPTIONS.items():
            print(f"{key}. {industry} - {description}")
        industry_choice = int(input("Select the industry by entering the corresponding number: "))
        industry = INDUSTRY_OPTIONS[industry_choice][0]

        # Duration
        y, sr = librosa.load(file_path)
        duration = librosa.get_duration(y=y, sr=sr)

        # New code to handle subcategories
        if industry in INDUSTRY_SUBCATEGORIES:
            print("Select a subcategory:")
            subcat_mapping = {}
            for idx, subcat in enumerate(INDUSTRY_SUBCATEGORIES[industry], 1):
                print(f"{chr(96 + idx)}. {subcat}")  # Maps 'a' to 1st option, 'b' to 2nd, etc.
                subcat_mapping[chr(96 + idx)] = subcat

            subcat_choice = input("Select the subcategory by entering the corresponding letter: ")
            industry_subcat = subcat_mapping.get(subcat_choice.strip(), "N/A")  # Default to "N/A" if invalid choice
        else:
            industry_subcat = "N/A"

        # New code for age demographics (Target Demographic)
        print("Select the target demographic:")
        for key, value in AGE_DEMOGRAPHICS.items():
            print(f"{key}. {value}")
        target_demographic_choice = int(input("Select the target demographic by entering the corresponding number: "))
        target_demographic = AGE_DEMOGRAPHICS[target_demographic_choice]
        
        voice_presence_human = input("Does the file contain voice? (y/n) ").lower()
        voice_type = ""
        age_range = ""
        character = ""
        gender = ""
        language = "None"
        region = "None"

        VO_attributes_list = []
        Vocal_attributes_list = []

        if voice_presence_human == "y":
            voice_type_choice = int(input(f"Is it Voice Over, Vocal, or Both? {', '.join([f'{k}. {v}' for k, v in VOICE_TYPES.items()])}: "))
            voice_type = VOICE_TYPES[voice_type_choice]
            
            # For handling multiple Voice Over artists
            if voice_type_choice in [1, 3]:  # Voice Over or Both
                multiple_VO = input("Are there multiple Voice Over artists? (y/n): ").lower() == 'y'
                if multiple_VO:
                    num_VO = int(input("How many Voice Over artists are there?: "))
                    for i in range(num_VO):
                        VO_age_range_choice = int(input(f"Age Range for Voice Over {i+1}? {', '.join([f'{k}. {v}' for k, v in AGE_DEMOGRAPHICS.items()])}: "))
                        VO_age_range = AGE_DEMOGRAPHICS[VO_age_range_choice]
                        VO_character_choice = int(input(f"Character for Voice Over {i+1}? {', '.join([f'{k}. {v}' for k, v in CHARACTERS.items()])}: "))
                        VO_character = CHARACTERS[VO_character_choice]
                        while True:
                            try:
                                VO_gender_choice = int(input(f"Gender for Voice Over {i+1}? {', '.join([f'{k}. {v}' for k, v in GENDERS.items()])}: "))
                                if VO_gender_choice in GENDERS.keys():
                                    VO_gender = GENDERS[VO_gender_choice]
                                    break
                                else:
                                    print("Invalid choice. Please enter 1 for Male or 2 for Female.")
                            except ValueError:
                                print("Invalid input. Please enter a number.")
                        VO_attributes_list.append(f"VO {i+1}: {AGE_DEMOGRAPHICS[VO_age_range_choice]}-{CHARACTERS[VO_character_choice]}-{GENDERS[VO_gender_choice]}")

                else:
                    age_range_choice = int(input(f"Age Range for Voice Over 1? {', '.join([f'{k}. {v}' for k, v in AGE_DEMOGRAPHICS.items()])}: "))
                    character_choice = int(input(f"Character for Voice Over 1? {', '.join([f'{k}. {v}' for k, v in CHARACTERS.items()])}: "))
                    gender_choice = int(input(f"Gender for Voice Over 1? {', '.join([f'{k}. {v}' for k, v in GENDERS.items()])}: "))
                    VO_attributes_list.append(f"VO 1: {AGE_DEMOGRAPHICS[age_range_choice]}-{CHARACTERS[character_choice]}-{GENDERS[gender_choice]}")

            # For handling multiple Vocal artists
            if voice_type_choice in [2, 3]:  # Vocal or Both
                multiple_Vocal = input("Are there multiple vocalists? (y/n): ").lower() == 'y'
                if multiple_Vocal:
                    num_Vocal = int(input("How many vocalists are there?: "))
                    for i in range(num_Vocal):
                        Vocal_age_range_choice = int(input(f"Age Range for Vocalist {i+1}? {', '.join([f'{k}. {v}' for k, v in AGE_DEMOGRAPHICS.items()])}: "))
                        Vocal_age_range = AGE_DEMOGRAPHICS[Vocal_age_range_choice]
                        Vocal_character_choice = int(input(f"Character for Vocalist {i+1}? {', '.join([f'{k}. {v}' for k, v in CHARACTERS.items()])}: "))
                        Vocal_character = CHARACTERS[Vocal_character_choice]
                        while True:
                            try:
                                Vocal_gender_choice = int(input(f"Gender for Vocalist {i+1}? {', '.join([f'{k}. {v}' for k, v in GENDERS.items()])}: "))
                                if Vocal_gender_choice in GENDERS.keys():
                                    Vocal_gender = GENDERS[Vocal_gender_choice]
                                    break
                                else:
                                    print("Invalid choice. Please enter 1 for Male or 2 for Female.")
                            except ValueError:
                                print("Invalid input. Please enter a number.")
                        Vocal_attributes_list.append(f"Vocalist {i+1}: {AGE_DEMOGRAPHICS[Vocal_age_range_choice]}-{CHARACTERS[Vocal_character_choice]}-{GENDERS[Vocal_gender_choice]}")

                else:
                    age_range_choice = int(input(f"Age Range for Vocalist 1? {', '.join([f'{k}. {v}' for k, v in AGE_DEMOGRAPHICS.items()])}: "))
                    character_choice = int(input(f"Character for Vocalist 1? {', '.join([f'{k}. {v}' for k, v in CHARACTERS.items()])}: "))
                    gender_choice = int(input(f"Gender for Vocalist 1? {', '.join([f'{k}. {v}' for k, v in GENDERS.items()])}: "))
                    Vocal_attributes_list.append(f"Vocalist 1: {AGE_DEMOGRAPHICS[age_range_choice]}-{CHARACTERS[character_choice]}-{GENDERS[gender_choice]}")
                
            language_choice = int(input(f"Language? {', '.join([f'{k}. {v}' for k, v in LANGUAGES.items()])}: "))
            language = LANGUAGES[language_choice]
            region_choice = int(input(f"Region? {', '.join([f'{k}. {v}' for k, v in REGIONS.items()])}: "))
            region = REGIONS[region_choice]

        # Confirmation step
        confirm = input("Are you satisfied with your labeling? (y/n): ").lower()
        if confirm == 'y':
            break  # Breaks out of the loop if the labeler is satisfied

    insights = {
        'Human Label': main_label,
        'Brand Name': brand_name,
        'Industry': industry,
        'Subcategory': industry_subcat,
        'Duration': duration,
        'Language': language,
        'Region': region,
        'Age Demographic': target_demographic,  # New column name for 'Age Demographic
        'Voice Presence': 'none' if voice_presence_human == 'n' else 'contains voice',
        'Voice Type': voice_type,
        'VO_Age Range': '; '.join(VO_attributes_list) if multiple_VO else (VO_attributes_list[0] if VO_attributes_list else ''),
        'VO_Character': '; '.join(VO_attributes_list) if multiple_VO else (VO_attributes_list[0] if VO_attributes_list else ''),
        'VO_Gender': '; '.join(VO_attributes_list) if multiple_VO else (VO_attributes_list[0] if VO_attributes_list else ''),
        'Vocal_Age Range': '; '.join(Vocal_attributes_list) if multiple_Vocal else (Vocal_attributes_list[0] if Vocal_attributes_list else ''),
        'Vocal_Character': '; '.join(Vocal_attributes_list) if multiple_Vocal else (Vocal_attributes_list[0] if Vocal_attributes_list else ''),
        'Vocal_Gender': '; '.join(Vocal_attributes_list) if multiple_Vocal else (Vocal_attributes_list[0] if Vocal_attributes_list else ''),
        'VO Attributes': '; '.join(VO_attributes_list),
        'Vocal Attributes': '; '.join(Vocal_attributes_list)
    }
    return insights

def batch_process(directory):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    audio_files = [f for f in all_files if f.lower().endswith(('.wav', '.mp3'))]

    #initialize advanced summary and basic summary
    basic_summary = ""
    named_entities = ""
    common_nouns = ""
    sentiment_score = ""
    
    documents_path = os.path.join(os.path.expanduser('~'), 'Documents')
    excel_path_xlsx = os.path.join(documents_path, 'Sonic Analysis Database.xlsx')
    excel_path_xls = os.path.join(documents_path, 'Sonic Analysis Database.xls')
    excel_path = excel_path_xlsx if os.path.exists(excel_path_xlsx) else excel_path_xls if os.path.exists(excel_path_xls) else excel_path_xlsx
    
    columns = [
        'Number', 'Date Added', 'File Name', 'Sonic Identifier', 'Ori/Dup', 'Human Label', 
        'Brand Name', 'Industry', 'Subcategory','Age Demographic', 'Duration', 'Language', 'Region',
        'Voice Presence', 'Voice Type', 'VO Artist', 'Vocal Artist',
        'Average Pitch', 'Sound Dynamics', 'Energy', 'Rhythmic Stability',
        'Tempo', 'Average Brightness', 'Musical Scale', 'avg_pitch_val', 'tempo_val',
        'spec_centroid_mean', 'max_db_voice', 'max_db_music', 'avg_db_voice', 'avg_db_music',
        'Basic Summary', 'named_entities', 'common_nouns', 'sentiment_score', 'acousticness', 'danceability',
        'energy', 'instrumentalness', 'loudness', 'valence'
    ]
    
    for i in range(13):  # 13 MFCCs
        for stat in ['Mean', 'Std', 'Min', 'Max', 'Median']:
            columns.append(f'MFCC_{i+1}_{stat}')

    df = pd.read_excel(excel_path) if os.path.exists(excel_path) else pd.DataFrame(columns=columns)
    identifier_to_file_index = dict(zip(df['Sonic Identifier'].tolist(), df.index))
    existing_identifiers = set(identifier_to_file_index.keys())
    
    for file_path in audio_files:
        try:
            # Call the function and store the results
            max_db_voice, max_db_music_and_elements, avg_db_voice, avg_db_music_and_elements = separate_and_measure_using_spleeter(file_path)
            audio_data = process_audio_file_updated(file_path)
            labels = play_audio_and_get_label(file_path)

            named_entities = None
            common_nouns_count = None
            sentiment_score = None
            basic_summary = None
            
            # If the audio contains voice according to human label
            if labels['Voice Presence'] == 'contains voice':
                # Transcribe the audio to text
                transcribed_text = transcribe_audio(file_path)
                
                if transcribed_text:
                    analytics = summarize_text_advanced(transcribed_text)
                    basic_summary = summarize_text(transcribed_text)
                    
                    named_entities = ', '.join(list(analytics['named_entities'].keys()))
                    common_nouns_count = ', '.join(f"{k}:{v}" for k, v in analytics['common_nouns'].items())
                    sentiment_score = analytics['sentiment_score']
                
                if avg_db_voice > avg_db_music_and_elements:
                    comparison = f"Voice higher by {abs(avg_db_voice - avg_db_music_and_elements):.2f} dB"
                else:
                    comparison = f"Music & Elements higher by {abs(avg_db_music_and_elements - avg_db_voice):.2f} dB"
            else:
                max_db_voice, max_db_music_and_elements, avg_db_voice, avg_db_music_and_elements = None, None, None, None
                comparison = None

            if audio_data['Sonic Identifier'] in existing_identifiers:
                original_index = identifier_to_file_index[audio_data['Sonic Identifier']]
                df.at[original_index, 'Ori/Dup'] = 'Ori'
                audio_data['Ori/Dup'] = 'Dup'
            else:
                audio_data['Ori/Dup'] = ''
                existing_identifiers.add(audio_data['Sonic Identifier'])
                identifier_to_file_index[audio_data['Sonic Identifier']] = len(df)
            if 'Avg_Pitch_Val' in audio_data:
                avg_pitch_val = audio_data['Avg_Pitch_Val']
            else:
                avg_pitch_val = None  # Or some default value
            if 'Tempo_Val' in audio_data:
                tempo_val = audio_data['Tempo_Val']
            else:
                tempo_val = None  # Or some default value
            if 'Spec_Centroid_Mean' in audio_data:
                spec_centroid_mean = audio_data['Spec_Centroid_Mean']
            else:
                spec_centroid_mean = None

            new_data = pd.DataFrame({
                'Number': [len(df) + 1],
                'Date Added': [datetime.now()],
                'Subcategory': [labels['Subcategory']],
                'VO Artist': [labels['VO Attributes']],  # New column name for 'VO Artist
                'Vocal Artist': [labels['Vocal Attributes']],  # New column name for 'Vocal Artist
                'Voice Presence': [labels['Voice Presence']],
                **labels,
                **audio_data,
                'avg_pitch_val': [audio_data['Avg_Pitch_Val']],
                'tempo_val': [audio_data['Tempo_Val']],
                'spec_centroid_mean': [audio_data['Spec_Centroid_Mean']],
                "avg_db_voice": [audio_data['avg_db_voice']],
                "avg_db_music": [audio_data['avg_db_music']],
                "max_db_voice": [audio_data['max_db_voice']],
                "max_db_music": [audio_data['max_db_music']],
                "Basic Summary": [basic_summary],
                "named_entities": [named_entities],
                "common_nouns": [common_nouns_count],
                "sentiment_score": [sentiment_score],
                "acousticness": [audio_data['acousticness']],
                "danceability": [audio_data['danceability']],
                "energy": [audio_data['energy']],
                "instrumentalness": [audio_data['instrumentalness']],
                "loudness": [audio_data['loudness']],
                "valence": [audio_data['valence']]
            })
            df = pd.concat([df, new_data], ignore_index=True)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    # Reorder columns
    columns_order = [
        'Number', 'Date Added', 'File Name', 'Sonic Identifier', 'Ori/Dup', 'Human Label', 
        'Brand Name', 'Industry', 'Subcategory','Age Demographic', 'Duration', 'Language', 'Region',
        'Voice Presence', 'Voice Type', 'VO Artist', 'Vocal Artist', 'Basic Summary', 'named_entities', 'common_nouns', 'sentiment_score',
        'Average Pitch', 'Sound Dynamics', 'Energy', 'Rhythmic Stability',
        'Tempo', 'Average Brightness', 'Musical Scale', 'avg_pitch_val', 'tempo_val', 'spec_centroid_mean',
        'max_db_voice', 'max_db_music', 'avg_db_voice', 'avg_db_music', 'acousticness', 'danceability', 'energy', 'instrumentalness',
        'loudness','valence'
    ]
    df = df[columns_order]
    df.to_excel(excel_path, index=False)

# Define the path to your directory of audio files
directory_path = "C:\\Users\\USER\\Documents\\Agson Sonics Python Codes\\Sonics Analysis Test Files"
batch_process(directory_path)