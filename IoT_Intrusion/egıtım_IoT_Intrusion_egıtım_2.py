

import warnings, joblib, shutil
from pathlib import Path
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# ─── Dosya Yolları ─────────────────────────────────────────────────────────
CSV_PATH = Path(r"C:\Users\İSO\Desktop\VERI_SETLERI\IoT_Intrusion\IoT_Intrusion_1.csv")
BASE_DIR = CSV_PATH.parent
PROC_DIR = BASE_DIR / "processed"; PROC_DIR.mkdir(exist_ok=True)
CM_DIR   = PROC_DIR / "model_cms"; CM_DIR.mkdir(exist_ok=True)
SEED = 42

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV bulunamadı: {CSV_PATH}")

# ─── 1. Veri Yükle & Etiket Encode ─────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print("Veri yüklendi:", df.shape)

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])
print("Etiket dönüşümü:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

X = df.drop(columns=["label"])
y = df['label']

# ─── 2. Train/Test Split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# Ölçekleyici (LR & KNN)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── 3. Model Havuzu ───────────────────────────────────────────────────────
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1, multi_class='multinomial'),
    "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "KNN":                KNeighborsClassifier(n_neighbors=5),
    "XGBoost":            XGBClassifier(n_estimators=600, max_depth=10, learning_rate=0.1,
                              subsample=0.9, colsample_bytree=0.9,
                              objective='multi:softprob', eval_metric='mlogloss',
                              random_state=SEED, n_jobs=-1)
}

summary = []

# ─── 4. Eğitim Döngüsü ─────────────────────────────────────────────────────
for name, model in MODELS.items():
    print(f"\n▶︎ {name} eğitiliyor…")

    if name in {"LogisticRegression", "KNN"}:
        model.fit(X_train_sc, y_train)
        preds = model.predict(X_test_sc)
        bundle = {"scaler": scaler, "clf": model}
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        bundle = {"scaler": None, "clf": model}

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average='macro')
    summary.append({"model": name, "accuracy": acc, "macro_f1": f1})

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Tahmin"); plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(CM_DIR / f"{name}_cm.png", dpi=300)
    plt.close()

    # Model kaydet
    joblib.dump(bundle, PROC_DIR / f"{name}.pkl")
    print(f"   ↳ {PROC_DIR}/{name}.pkl (F1={f1:.3f})")

# ─── 5. En İyi Model & Özet ────────────────────────────────────────────────
summary_df = pd.DataFrame(summary).sort_values("macro_f1", ascending=False)
summary_df.to_csv(BASE_DIR / "iot_intrusion_models_summary.csv", index=False)

best_name = summary_df.iloc[0]['model']
shutil.copy(PROC_DIR / f"{best_name}.pkl", PROC_DIR / "best_model.pkl")
print(f"\n🏆 En iyi model: {best_name} (F1={summary_df.iloc[0]['macro_f1']:.3f}) → {PROC_DIR}/best_model.pkl")

print("\n✅ Eğitim tamamlandı – Sonuçlar:")
print(summary_df)
print(f"Tüm çıktı → {PROC_DIR}")
