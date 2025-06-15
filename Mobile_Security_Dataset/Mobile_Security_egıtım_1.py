# prepare_mobile_dataset.py
# ───────────────────────────────────────────────────────────────
"""
Mobile Security Dataset • Tanıma → Görselleştirme → Temizleme Pipeline
Çalıştır:  python prepare_mobile_dataset.py
"""

import os, datetime
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Ayarlar ────────────────────────────────────────────────────
FOLDER      = r"C:\Users\İSO\Desktop\VERI_SETLERI\Mobile_Security_Dataset"
RAW_CSV     = os.path.join(FOLDER, "Mobile Security Dataset.csv")
CLEAN_CSV   = os.path.join(FOLDER, "Mobile_Security_Dataset_1.csv")
LOG_TXT     = os.path.join(FOLDER, "degisen.txt")
PLOT_DIR    = os.path.join(FOLDER, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

# ─── 1) Veri Yükle ──────────────────────────────────────────────
df = pd.read_csv(RAW_CSV, encoding="utf-8")
orig_shape = df.shape

# ─── 2) Keşifsel Analiz & Görseller ────────────────────────────
def save_plot(fig, fname):
    path = os.path.join(PLOT_DIR, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

log_lines = [
    f"=== Mobile Security Dataset – Hazırlık Raporu ===",
    f"Tarih: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
    f"Ham boyut: {orig_shape[0]} satır × {orig_shape[1]} sütun\n"
]

# Kategorilerin dağılımı
fig, ax = plt.subplots(figsize=(10,6))
df["Category"].value_counts().plot.bar(ax=ax)
ax.set_title("Kategori Dağılımı")
ax.set_ylabel("Adet")
save_plot(fig, "kategori_dagilimi.png")

# Eksik değer ısı haritası
fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(df.isna(), cbar=False, ax=ax)
ax.set_title("Eksik Değer Isı Haritası")
save_plot(fig, "missing_heatmap.png")

# Metin sütunlarının uzunluk histogramı
TEXT_COLS = [
    "Security_Practice_Used","Vulnerability_Types","Mitigation_Strategies",
    "Developer_Challenges","Assessment_Tools_Used","Improvement_Suggestions"
]
df["text_len"] = df[TEXT_COLS].fillna("").agg(" ".join, axis=1).str.len()
fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df["text_len"], bins=40, ax=ax)
ax.set_title("Bileşik Metin Uzunluğu")
save_plot(fig, "metin_uzunlugu.png")

# ─── 3) Temizleme / Dönüştürme ─────────────────────────────────
changes = []

# 3.1 Sütun adlarındaki baş/son boşluklar
strip_before = df.columns.tolist()
df.columns = df.columns.str.strip()
if df.columns.tolist() != strip_before:
    changes.append("Sütun adlarındaki boşluklar kaldırıldı")

# 3.2 Sembolik eksik değerler → NaN
missing_before = (df == "?").sum().sum() + (df == "").sum().sum()
df.replace(["?", ""], np.nan, inplace=True)
missing_after = df.isna().sum().sum()
if missing_after > 0:
    changes.append(f'"?"/"" sembolleri NaN\'e çevrildi (toplam {missing_before} hücre)')

# 3.3 Eksik hücreleri “Unknown” ile doldur
df.fillna("Unknown", inplace=True)
changes.append(f"Tüm NaN hücreler 'Unknown' ile dolduruldu (şu an eksik değer yok)")

# 3.4 Az örnekli sınıfları at ( < 100 )
class_counts = df["Category"].value_counts()
rare_classes = class_counts[class_counts < 100].index
if len(rare_classes):
    before_rows = len(df)
    df = df[~df["Category"].isin(rare_classes)].reset_index(drop=True)
    removed = before_rows - len(df)
    changes.append(f"Az örnekli {len(rare_classes)} sınıf çıkarıldı, {removed} satır silindi")

# 3.5 Geçici sütun temizliği
df.drop(columns=["text_len"], inplace=True, errors="ignore")

log_lines.append(">>> Temizleme Adımları:")
log_lines.extend(f"- {c}" for c in changes)
log_lines.append("")
log_lines.append(f"Temiz boyut: {df.shape[0]} satır × {df.shape[1]} sütun")

# ─── 4) Sonuçları Kaydet ───────────────────────────────────────
df.to_csv(CLEAN_CSV, index=False, encoding="utf-8")
log_lines.append(f"\nTemiz veri kaydedildi → {os.path.basename(CLEAN_CSV)}")

# 4.1 Değişim raporu
with open(LOG_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

print("✅ İşlem tamamlandı.")
print(f"• Görseller: {PLOT_DIR}")
print(f"• Temiz CSV: {CLEAN_CSV}")
print(f"• Değişim raporu: {LOG_TXT}")
