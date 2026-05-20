import pandas as pd
import os

file_csv = 'data_dashboard_with_date.csv'

if not os.path.exists(file_csv):
    print(f"❌ File '{file_csv}' TIDAK DITEMUKAN.")
    exit()

df = pd.read_csv(file_csv)

print("====================================")
print("🔎 CEK KOLOM KRITIS DI DATA HISTORY")
print("====================================")
print("Kolom yang tersedia di file CSV:")
print(df.columns.tolist())

print("\n--- STATUS KOLOM ---")

if 'Tanggal' in df.columns:
    print("✅ 'Tanggal': Ditemukan.")
else:
    print("❌ 'Tanggal': TIDAK DITEMUKAN. (Harus ada untuk sumbu X)")

if 'Sentimen_Prediksi' in df.columns:
    print("✅ 'Sentimen_Prediksi': Ditemukan.")
    # Cek apakah ada nilai 0 dan 1 (Negatif/Positif)
    counts = df['Sentimen_Prediksi'].value_counts()
    print(f"   Nilai unik: {counts.to_dict()}")
    
    if counts.get(0, 0) == 0 and counts.get(1, 0) == 0:
        print("   ⚠️ Peringatan: Kolom Sentimen_Prediksi ada, tetapi semua baris mungkin kosong/NaN.")
        
else:
    print("❌ 'Sentimen_Prediksi': TIDAK DITEMUKAN! INI PENYEBAB UTAMA. (Harus ada nilai 0/1)")
    print("   Solusi: Ganti nama kolom prediksi di CSV Anda menjadi 'Sentimen_Prediksi'.")