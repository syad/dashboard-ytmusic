# =======================
#  PREPROCESSING MODULE
#  File: preprocessing.py
# =======================

import re
import string
import io
import requests
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Pastikan resource NLTK tersedia
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
except:
    pass

# Setup Stemmer Sastrawi
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    SASTRAWI_AVAILABLE = True
except Exception:
    stemmer = None
    SASTRAWI_AVAILABLE = False
    print("Warning: Sastrawi tidak ditemukan. Stemming tidak akan berjalan.")

# URL Kamus Alay/Slang
LEXICON_URL = "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"

# =============================
# LOAD SLANG LEXICON
# =============================
def load_lexicon(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        df_lexicon = pd.read_csv(io.StringIO(response.text))
        slang_dict = dict(zip(df_lexicon["slang"], df_lexicon["formal"]))
    except Exception:
        slang_dict = {}

    # Tambahan slang internal (Custom Qpon)
    custom_slang_additions = {
        'bgt': 'banget', 'yg': 'yang', 'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak',
        'ngga': 'tidak', 'nggak': 'tidak', 'engga': 'tidak', 'enggak': 'tidak','gaada': 'tidak ada', 'utk': 'untuk',
        'ok': 'oke', 'okeey': 'oke', 'thanks': 'terima kasih', 'thx': 'terima kasih',
        'sih': '', 'dong': '', 'deh': '', 'nih': '', 'kok': '', 'yaa': 'ya',
        'aja': 'saja', 'nyari': 'mencari', 'bikin': 'membuat', 'apk': 'aplikasi',
        'app': 'aplikasi', 'mulu': 'melulu', 'gabisa': 'tidak bisa', 'gbisa': 'tidak bisa',
        'dgn': 'dengan', 'lemot': 'lambat', 'error': 'error', 'crash': 'crash',
        'makasih': 'terima kasih', 'mksih': 'terima kasih', 'knp': 'kenapa',
        'parah': 'parah', 'mantap': 'mantap', 'keren': 'keren', 'jgn': 'jangan',
        'sdh': 'sudah', 'blm': 'belum'
    }
    slang_dict.update(custom_slang_additions)
    return slang_dict

# =============================
# LOAD STOPWORDS
# =============================
def get_stopwords_id():
    try:
        base = stopwords.words("indonesian")
    except:
        base = []
        
    custom = [
        "qpon", "aplikasi", "nya", "sih", "ya", "yg", "saja", "aja", "mulu", "dong",
        "deh", "kak", "min", "admin", "gan", "bro", "nih","baiknya", "berkali", "kali", "kurangnya", "mata", 
        "olah", "sekurang", "setidak", "tama", "tidaknya"
    ]
    base.extend(custom)
    return set(base)

# Inisialisasi Variabel Global
SLANG_DICT = load_lexicon(LEXICON_URL)
STOPWORDS_ID = get_stopwords_id()

# =============================
# MAIN PREPROCESSING FUNCTION
# =============================
def preprocess_text(text):
    """
    Pipeline lengkap untuk preprocessing 1 teks ulasan.
    Menghapus URL, angka, tanda baca, stopwords, slang normalization, tokenisasi,
    dan stemming dengan Sastrawi jika tersedia.
    """

    if not isinstance(text, str):
        return ""

    # 1. Lowercase & Cleaning
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Hapus URL
    text = re.sub(r"\@\w+|\#\w+", "", text)              # Hapus mention/hashtag
    text = re.sub(r"\d+", "", text)                      # Hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation)) # Hapus tanda baca
    text = text.strip()
    text = re.sub(r"\s+", " ", text)                     # Hapus spasi ganda

    # 2. Slang Normalization
    words = text.split()
    normalized_words = [SLANG_DICT.get(word, word) for word in words]
    text = " ".join(normalized_words)

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Stopword Removal
    cleaned_tokens = [w for w in tokens if w not in STOPWORDS_ID and len(w) > 2]
    text_clean = " ".join(cleaned_tokens)

    # 5. Stemming (Sastrawi)
    if SASTRAWI_AVAILABLE and stemmer:
        return stemmer.stem(text_clean)
    
    return text_clean

# --- Alias agar kompatibel dengan app.py sebelumnya ---
clean_and_preprocess = preprocess_text

# =============================
# HELPER FOR DATAFRAME
# =============================
def process_dataframe(df, text_column='text', new_column='text_clean'):
    """
    Fungsi bantuan untuk memproses satu kolom DataFrame sekaligus.
    Berguna untuk fitur 'Upload CSV' di Streamlit.
    """
    if text_column not in df.columns:
        return df
    
    # Gunakan progress bar jika library tqdm tersedia (opsional)
    try:
        from tqdm import tqdm
        tqdm.pandas()
        df[new_column] = df[text_column].progress_apply(preprocess_text)
    except ImportError:
        df[new_column] = df[text_column].apply(preprocess_text)
        
    return df

# =============================
# TESTING BLOCK
# =============================
if __name__ == "__main__":
    # Blok ini hanya jalan jika file dijalankan langsung (python preprocessing.py)
    # Gunanya untuk testing sebelum dipakai di app.py
    
    sample_text = "Apk nya lemot bgt gan, tlg diperbaiki dong! @admin 123 https://qpon.com"
    print("--- Test Preprocessing ---")
    print(f"Original: {sample_text}")
    print(f"Hasil   : {preprocess_text(sample_text)}")
    
    if SASTRAWI_AVAILABLE:
        print("Status  : Sastrawi Aktif ✅")
    else:
        print("Status  : Sastrawi Tidak Aktif ❌ (Stemming dilewati)")
