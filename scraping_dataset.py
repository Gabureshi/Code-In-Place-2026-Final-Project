#download necessary libraries from requirements.txt

# Mengimpor pustaka google_play_scraper untuk mengakses ulasan dan informasi aplikasi dari Google Play Store.
from google_play_scraper import app, reviews, Sort, reviews_all

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
seed = 42
np.random.seed(seed)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import datetime as dt
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from wordcloud import WordCloud

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

import time
import csv
import requests
from io import StringIO

# ==================== SCRAPING WITH PROGRESS ====================

def scrape_reviews_with_progress(app_id, max_reviews=19000, batch_size=199):
    """
    Scrape reviews with progress tracking
    """
    all_reviews = []
    continuation_token = None
    total_collected = 0
    
    print(f"Starting to scrape reviews for {app_id}")
    print(f"Target: {max_reviews} reviews")
    print("-" * 50)
    
    while total_collected < max_reviews:
        remaining = max_reviews - total_collected
        current_batch = min(batch_size, remaining)
        
        try:
            result, continuation_token = reviews(
                app_id,
                lang='id',
                country='id',
                sort=Sort.MOST_RELEVANT,
                count=current_batch,
                continuation_token=continuation_token
            )
            
            if not result:
                print(f"\nNo more reviews available. Stopping at {total_collected} reviews.")
                break
                
            all_reviews.extend(result)
            total_collected += len(result)
            
            progress = (total_collected / max_reviews) * 100
            print(f"Progress: {progress:.1f}% | Collected: {total_collected}/{max_reviews} reviews | "
                  f"Batch: {len(result)} reviews")
            
            time.sleep(0.5)
            
            if not continuation_token:
                print(f"\nReached end of available reviews. Total collected: {total_collected}")
                break
                
        except Exception as e:
            print(f"Error occurred: {e}")
            break
    
    print("-" * 50)
    print(f"✅ Scraping completed! Total reviews collected: {len(all_reviews)}")
    
    return all_reviews

# Perform scraping
scrapreview = scrape_reviews_with_progress('com.mobile.legends', max_reviews=19000)

# Create DataFrame
app_reviews_df = pd.DataFrame(scrapreview)

# Save raw data
app_reviews_df.to_csv('ulasan_aplikasi.csv', index=False)
print(f"\n✅ Raw data saved to 'ulasan_aplikasi.csv'")

# ==================== DATA CLEANING ====================

# Display basic info
print(f"\nData shape: {app_reviews_df.shape}")
print(f"Columns: {app_reviews_df.columns.tolist()}")

# Remove nulls and duplicates
clean_df = app_reviews_df.dropna()
clean_df = clean_df.drop_duplicates()
print(f"After cleaning: {clean_df.shape}")

# ==================== TEXT PREPROCESSING FUNCTIONS ====================

def cleaningText(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')
    return text

def casefoldingText(text):
    return text.lower()

def tokenizingText(text):
    return word_tokenize(text)

def filteringText(text):
    listStopwords = set(stopwords.words('indonesian'))
    listStopwords1 = set(stopwords.words('english'))
    listStopwords.update(listStopwords1)
    listStopwords.update([
        # Filler words (no meaning)
        "yg", "nya", "sih", "dong", "kok", "deh", "yah", "ya", 
        "woi", "woy", "lah", "dah", "nih", "nah", "loh", "lho", 
        "si", "para", "pada",
        
        # Conjunctions & prepositions (neutral)
        "dan", "atau", "tapi", "namun", "sedangkan", "karena",
        "jika", "maka", "maka", "sehingga", "agar", "supaya",
        
        # Pronouns (neutral)
        "saya", "aku", "kamu", "dia", "mereka", "kami", "kita",
        "anda", "beliau", "nya", "ku", "mu", "kau",
        
        # Game identifiers (neutral - just naming the game)
        "game", "ml", "mlbb", "mobile", "legends", 
        "mobilelegends", "mlb", "wildrift", "app", "aplikasi",
        
        # Technical neutral terms (features without sentiment)
        "mode", "classic", "season", "event", "map", 
        "matchmaking", "network", "version",
        
        # Generic references (neutral)
        "akun", "id", "username", "nickname", "nama", "user",
        "player", "enemy", "teman",
        
        # Verb helpers (neutral)
        "pakai", "guna", "make", "mainnya",
        
        # Time references (context-dependent but often neutral)
        "hari", "bulan", "tahun", "kemarin", "besok", "sekarang", "nanti",
    ])
    return [txt for txt in text if txt not in listStopwords]

def stemmingText(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def toSentence(list_words):
    if not list_words:
        return ""
    return ' '.join(list_words)

# Slang dictionary
slangwords = {
    # Profanity (maps to negative sentiment words)
    "anjir": "anjing",
    "anjrit": "anjing", 
    "anjing": "anjing",
    "bangke": "bangkai",
    "bangsat": "bangsat",
    "bego": "bodoh",
    "goblok": "bodoh", 
    "tolol": "bodoh",
    "idiot": "bodoh",
    "bodoh": "bodoh",
    "sialan": "sial",
    "ampas": "ampas",
    "cacat": "cacat",
    "setan": "setan",
    "neraka": "neraka",
    
    # ===== PERFORMANCE ISSUES (Negative) =====
    "lemot": "lemot",
    "lelet": "lelet", 
    "lag": "lag",
    "bug": "bug",
    "error": "error",
    "crash": "crash",
    "freeze": "freeze",
    "dc": "disconnect",
    "putus": "disconnect",
    
    # ===== POSITIVE SLANG =====
    "gacor": "gacor",
    "candu": "candu",
    "ketagihan": "ketagihan",
    "asik": "asik",
    "asyik": "asik",
    "kocak": "kocak",
    "gilaa": "gila",
    "keren": "keren",
    "keren abis": "keren",
    "mantap": "mantap",
    "jos": "jos",
    "gass": "gas",
    
    # ===== NEUTRAL GAMING TERMS (Keep as-is) =====
    # Game modes & features
    "rank": "rank",
    "ranked": "rank",
    "classic": "classic",
    "brawl": "brawl",
    "arcane": "arcane",
    "mayhem": "mayhem",
    "survival": "survival",
    
    # Game mechanics
    "hero": "hero",
    "heroes": "hero",
    "skin": "skin",
    "skins": "skin",
    "emote": "emote",
    "emotes": "emote",
    "spell": "spell",
    "battle": "battle",
    "spell": "spell",
    
    # Actions
    "push": "push",
    "def": "defend",
    "defend": "defend",
    "attack": "attack",
    "gank": "gank",
    "rotate": "rotate",
    "split": "split",
    "farm": "farm",
    "jungle": "jungle",
    "lord": "lord",
    "turtle": "turtle",
    
    # Roles (neutral)
    "tank": "tank",
    "fighter": "fighter",
    "assassin": "assassin",
    "mage": "mage",
    "marksman": "marksman",
    "support": "support",
    "roamer": "roamer",
    "carry": "carry",
    
    # Communication (neutral)
    "report": "report",
    "reported": "report",
    "block": "block",
    "mute": "mute",
    "unmute": "unmute",
    
    # Status (neutral)
    "win": "win",
    "lose": "lose",
    "draw": "draw",
    "kalah": "kalah",
    "menang": "menang",
    "seri": "seri",
    
    # Time/rate (neutral)
    "mulu": "terus",
    "melulu": "terus",
    "terus": "terus",
    "trus": "terus",
    
    # ===== NEUTRAL SLANG & ABBREVIATIONS =====
    # Common gaming abbreviations
    "op": "overpower",
    "op banget": "overpower",
    "meta": "meta",
    "nerf": "nerf",
    "buff": "buff",
    "revamp": "revamp",
    "rework": "rework",
    
    # Casual terms (neutral)
    "si": "si",
    "para": "para",
    "nih": "nih",
    "nah": "nah",
    "loh": "loh",
    "dong": "dong",
    "sih": "sih",
    "kok": "kok",
    "deh": "deh",
    "yah": "yah",
    "ya": "ya",
    
    # Greetings/gaming chat (neutral)
    "hai": "hai",
    "hello": "hello",
    "halo": "halo",
    "gws": "good game",
    "gg": "good game",
    "wp": "well played",
    "glhf": "good luck have fun",
    
    # Response/acknowledgment (neutral)
    "oke": "oke",
    "ok": "oke",
    "iy": "iya",
    "iya": "iya",
    "ga": "tidak",
    "ngga": "tidak",
    "engga": "tidak",
    "tdk": "tidak",
    
    # ===== DIFFICULTY & QUANTITY =====
    "gampang": "mudah",
    "susah": "sulit",
    "bablas": "kalah",
    "dikit": "sedikit",
    "dikitt": "sedikit",
    "sdkit": "sedikit",
    "banyak": "banyak",
    "byk": "banyak",
    
    # ===== INTENSIFIERS =====
    "bgtt": "banget",
    "bgtu": "banget",
    "bgt": "banget",
    "banget": "banget",
    "sekali": "sekali",
    "skali": "sekali",
    "sangat": "sangat",
    "amat": "amat",
    "terlalu": "terlalu",
    
    # ===== COMMON TYPOS =====
    "bs": "bisa",
    "bsa": "bisa",
    "bisa": "bisa",
    "jg": "juga",
    "jga": "juga",
    "jga": "juga",
    "juga": "juga",
    "sdh": "sudah",
    "udah": "sudah",
    "blm": "belum",
    "belum": "belum",
    
    # ===== SOCIAL TERMS (Neutral) =====
    "wkwk": "wkwk",
    "wkwkwk": "wkwk",
    "wkwkwkwk": "wkwk",
    "gas": "gas",
    "gaskeun": "gas",
    "mabar": "mabar",
    "ngab": "ngab",
    "bro": "bro",
    "broo": "bro",
    "brother": "bro",
    "sist": "sist",
    "sister": "sist",
    "guys": "guys",
    "gan": "gan",
    "om": "om",
    "tante": "tante",
    
    # ===== EXCLAMATIONS (Neutral/Varies) =====
    "wah": "wah",
    "wow": "wow",
    "aduh": "aduh",
    "astaga": "astaga",
    "gile": "gila",
    "gilaa": "gila",
    "cuak": "cuak",
    "cip": "cip",
    "cakep": "cakep",
    "kece": "kece",
}


def fix_slangwords(text):
    words = text.split()
    fixed_words = [slangwords.get(word.lower(), word) for word in words]
    return ' '.join(fixed_words)

# ==================== APPLY PREPROCESSING ====================

print("\n" + "="*50)
print("TEXT PREPROCESSING")
print("="*50)

clean_df['text_clean'] = clean_df['content'].apply(cleaningText)
print("✓ Cleaning completed")

clean_df['text_casefoldingText'] = clean_df['text_clean'].apply(casefoldingText)
print("✓ Case folding completed")

clean_df['text_slangwords'] = clean_df['text_casefoldingText'].apply(fix_slangwords)
print("✓ Slang words fixed")

clean_df['text_tokenizingText'] = clean_df['text_slangwords'].apply(tokenizingText)
print("✓ Tokenization completed")

clean_df['text_stopword'] = clean_df['text_tokenizingText'].apply(filteringText)
print("✓ Stopwords removed")

clean_df['text_akhir'] = clean_df['text_stopword'].apply(toSentence)
print("✓ Final text prepared")

# Remove any empty strings that resulted from preprocessing
clean_df = clean_df[clean_df['text_akhir'].str.strip().astype(bool)]
print(f"After removing empty texts: {clean_df.shape}")

# ==================== SENTIMENT LABELING ====================

print("\n" + "="*50)
print("SENTIMENT LABELING WITH LEXICON")
print("="*50)

# Load lexicons
lexicon_positive = {}
response = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv')
if response.status_code == 200:
    reader = csv.reader(StringIO(response.text), delimiter=',')
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])
    print(f"✓ Positive lexicon loaded: {len(lexicon_positive)} words")
else:
    print("Failed to fetch positive lexicon data")

lexicon_negative = {}
response = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv')
if response.status_code == 200:
    reader = csv.reader(StringIO(response.text), delimiter=',')
    for row in reader:
        lexicon_negative[row[0]] = int(row[1])
    print(f"✓ Negative lexicon loaded: {len(lexicon_negative)} words")
else:
    print("Failed to fetch negative lexicon data")

def sentiment_analysis_lexicon_indonesia(text):
    score = 0
    for word in text:
        if word in lexicon_positive:
            score += lexicon_positive[word]
        if word in lexicon_negative:
            score += lexicon_negative[word]
    
    if score > 0:
        polarity = 'positive'
    elif score < 0:
        polarity = 'negative'
    else:
        polarity = 'neutral'
    
    return score, polarity

# Apply sentiment analysis
results = clean_df['text_stopword'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
clean_df['polarity_score'] = results[0]
clean_df['polarity'] = results[1]

# Display results
print("\n" + "="*50)
print("SENTIMENT DISTRIBUTION")
print("="*50)
print(clean_df['polarity'].value_counts())
print("\nPercentage:")
print(clean_df['polarity'].value_counts(normalize=True) * 100)

# ==================== SAVE CLEAN DATA ====================

clean_df.to_csv('clean_data.csv', index=False)
print("\n✅ Clean data saved to 'clean_data.csv'")
print(f"Total samples: {len(clean_df)}")
print(f"Columns saved: {clean_df.columns.tolist()}")

# ==================== BASIC STATISTICS ====================

print("\n" + "="*50)
print("DATASET STATISTICS")
print("="*50)
print(f"Total reviews scraped: {len(scrapreview)}")
print(f"After cleaning: {len(clean_df)}")
print(f"Removed: {len(scrapreview) - len(clean_df)} reviews ({(len(scrapreview)-len(clean_df))/len(scrapreview)*100:.1f}%)")
print(f"\nAverage review length: {clean_df['text_akhir'].str.len().mean():.1f} characters")
print(f"Median review length: {clean_df['text_akhir'].str.len().median():.1f} characters")