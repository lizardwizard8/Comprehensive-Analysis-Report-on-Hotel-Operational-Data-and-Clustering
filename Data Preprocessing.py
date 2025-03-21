import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from pycaret.clustering import *
import seaborn as sns

# Dataseti içe aktar
# (Dosya yolunu kendi bilgisayarına göre değiştirmen gerekebilir)
df = pd.read_excel("C:\\Users\\gunal\\Desktop\\Bitirme Projesi\\hotel_dataset_with_reformed_cooling.xlsx")

# 1. Tarih Kolonlarını İşle

# "Check-in Date" kolonunu düzenleyip ay ve haftanın günü olarak çıkartıyoruz.
if "Check-in Date" in df.columns:
    df["Check-in Date"] = pd.to_datetime(df["Check-in Date"], errors='coerce')
    df["Check-in Month"] = df["Check-in Date"].dt.month  # Ayı çıkar
    df["Check-in Weekday"] = df["Check-in Date"].dt.weekday  # Haftanın günü çıkar
    df.drop(["Check-in Date"], axis=1, inplace=True)  # Orijinal kolon işimize yaramayacağı için siliyoruz


# 2. Diğer Kolonları İşle

# Her kolonu işleyip sayısal hale getirmeye çalışacağız.
threshold = 10  # Eğer bir kategorik kolon 10'dan az unique değere sahipse one-hot encoding yapacağız.

df_processed = df.copy()
columns_to_drop = []  # One-hot encoding yapılan kolonları burada saklayacağız.

for col in df_processed.columns:
    if df_processed[col].dtype == 'object':
        df_processed[col] = df_processed[col].fillna("Eksik")  # Boş olan yerleri Eksik ile dolduruyoruz.
        
        # Eğer sayıya çevrilebiliyorsa çevir (Bazen sayılar string olarak saklanıyor, düzeltelim)
        try:
            df_processed[col] = pd.to_numeric(df_processed[col])
            continue  # Eğer dönüşüm başarılıysa ekstra işleme gerek yok
        except:
            pass  # Dönüşüm başarısız olduysa, kategorik olarak işlemeye devam edeceğiz

        unique_count = df_processed[col].nunique()
        
        # Eğer kolonun eşsiz değer sayısı azsa (10 veya daha az), one-hot encoding uygula
        if unique_count <= threshold:
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            columns_to_drop.append(col)  # Orijinal kolon artık gereksiz, sonradan sileceğiz
        else:
            # Eğer eşsiz değer sayısı fazlaysa label encoding kullan
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])

# One-hot encoding uygulanan orijinal kolonları sil
df_processed.drop(columns=columns_to_drop, inplace=True)


# 3. Eksik Sayısal Değerleri Doldur Eksik sayısal yok diye gördüm ama olsun
# Eksik olan sayısal değerleri median (ortanca) ile dolduruyoruz.
numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())


# 4. Sayısal Verileri Standartlaştır

# Verileri standardize etmek için StandardScaler kullanıyoruz.
scaler = StandardScaler()
df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])


# 5. Korelasyon Haritası

plt.figure(figsize=(12, 8))
sns.heatmap(df_processed[numeric_cols].corr(), cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Özellik Korelasyon Isı Haritası")
plt.show()

# 6. Son Haline Bakalım

print(df_processed.head())
print(df_processed.info())


# 7. PyCaret Clustering Ortamını Hazırla
exp_clustering = setup(data=df_processed, normalize=True, session_id=123)
