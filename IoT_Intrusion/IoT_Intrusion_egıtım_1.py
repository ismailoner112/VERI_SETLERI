import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

dosya_yolu = r"C:\Users\İSO\Desktop\VERI_SETLERI\IoT_Intrusion\IoT_Intrusion.csv"
df = pd.read_csv(dosya_yolu, low_memory=False)
df.columns = df.columns.str.strip()  # Sütun adlarını temizle


print(" Veri seti boyutu:", df.shape)
print("\n İlk 5 satır:\n", df.head())
print("\n Veri türleri:\n", df.dtypes.value_counts())
print("\n Eksik değer sayısı (varsa):\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\n Etiket dağılımı:\n", df['label'].value_counts())

# Etiket (label) görselleştirme
plt.figure(figsize=(12, 6))
sns.countplot(y=df['label'], order=df['label'].value_counts().index, palette="mako")
plt.title("IoT Intrusion Veri Etiketi Dağılımı")
plt.xlabel("Örnek Sayısı")
plt.ylabel("Etiket")
plt.tight_layout()
plt.savefig("IoT_label_dagilimi.png")
plt.show()

# Sorunlu değerleri '?' ve sonsuzlukları temizle
df.replace(['?', 'Infinity', '-Infinity', np.inf, -np.inf], np.nan, inplace=True)

#Sayısal olmayan ve modellemeye dahil edilmeyecek sütunları tespit et
non_numeric = df.select_dtypes(include='object').columns.drop('label', errors='ignore')
print("\n🧹 Model dışı sayısal olmayan sütunlar:", list(non_numeric))

#Sayısal olmayan sütunları kaldır
df.drop(columns=non_numeric, inplace=True)

# Eksik değerleri içeren satırları çıkar
df.dropna(inplace=True)

# Yeni temizlenmiş dosya adını belirle (eskinin üzerine yazmaz!)
temiz_yol = dosya_yolu.replace(".csv", "_1.csv")
df.to_csv(temiz_yol, index=False)

print(f"\n Temizlenmiş veri seti başarıyla kaydedildi:\n{temiz_yol}")

df.to_csv(r"C:\Users\İSO\Desktop\VERI_SETLERI\IoT_Intrusion\IoT_Intrusion_1.csv", index=False)
print("Temiz veri 'IoT_Intrusion_1.csv' olarak kaydedildi.")
