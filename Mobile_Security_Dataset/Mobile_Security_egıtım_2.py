import os, re, warnings, pickle, shutil, datetime
from pathlib import Path
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

# ─── STOP WORDS -------------------------------------------------------------
try:
    import nltk; nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    TR_STOP = set("""acaba ama ancak aslında az bazı belki beri bu biz çok çünkü da daha de defa diğer diye için ile ise gibi hem hiç yine hep hangi her hiçbir iken ile ilgili kez ki kim kimi kimse madem mi mu mü nasıl ne neden nedenle nerde nerede nereye niçin niye o öyle aynı pek sen siz şu şimdi sonra söz şu yüzden tüm üzere üstelik var ve veya veyahut ya yani yalnız yine yok""".split())
    STOP_WORDS = TR_STOP.union(stopwords.words('english'))
except Exception:
    STOP_WORDS = None

# ─── YOL SABİTLERİ ----------------------------------------------------------
DATA_DIR = Path(r"C:\Users\İSO\Desktop\VERI_SETLERI\Mobile_Security_Dataset")
CSV_PATH = DATA_DIR / "Mobile_Security_Dataset_1.csv"
PROC_DIR = DATA_DIR / "processed"; PROC_DIR.mkdir(exist_ok=True)
CM_DIR   = PROC_DIR / "model_cms"; CM_DIR.mkdir(exist_ok=True)
LOG_FILE = DATA_DIR / "degisim_v4.txt"
SEED     = 42

# ─── 1. Veri Yükle ----------------------------------------------------------
if not CSV_PATH.exists():
    raise FileNotFoundError(CSV_PATH)

df = pd.read_csv(CSV_PATH, encoding='utf-8')
print("Veri boyutu:", df.shape)

# ─── 2. Az Örnekli Sınıfları At -------------------------------------------
class_counts = df['Category'].value_counts()
small = class_counts[class_counts < 10]
with open(LOG_FILE, "w", encoding="utf-8") as lg:
    lg.write(f"Log oluşturma: {datetime.datetime.now()}\n")
    lg.write(class_counts.to_string())
    lg.write("\n\n")
    if not small.empty:
        lg.write("<10 kayıt sınıflar silindi:\n")
        lg.write(small.to_string())

df = df[~df['Category'].isin(small.index)].reset_index(drop=True)

# ─── 3. Metin Temizleme -----------------------------------------------------
COLS = ["Security_Practice_Used","Vulnerability_Types","Mitigation_Strategies",
        "Developer_Challenges","Assessment_Tools_Used","Improvement_Suggestions"]

def clean(s: str):
    s = str(s).lower()
    s = re.sub(r"[^\w\s]"," ",s)
    s = re.sub(r"\d+"," ",s)
    s = re.sub(r"_"," ",s)
    s = re.sub(r"\s+"," ",s).strip()
    return s

for c in COLS:
    if c in df.columns:
        df[c] = df[c].fillna("").map(clean)
    else:
        df[c] = ""

df['text'] = df[COLS].agg(" ".join, axis=1)

# ─── 4. Train/Test Split ----------------------------------------------------
X_train_txt, X_test_txt, y_train_str, y_test_str = train_test_split(
    df['text'].tolist(), df['Category'], test_size=0.2, stratify=df['Category'], random_state=SEED)

le = LabelEncoder(); y_train = le.fit_transform(y_train_str); y_test = le.transform(y_test_str)
CLASS_NAMES = le.classes_

# ─── 5. Yardımcı: Confusion Matrix -----------------------------------------

def save_cm(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(title); plt.xlabel('Tahmin'); plt.ylabel('Gerçek'); plt.tight_layout()
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.savefig(fname, dpi=300); plt.close()

summary = []

# ─── 6. TF‑IDF + Dengeli SVM ----------------------------------------------
print("▶︎ TF‑IDF + SVM")
vec = TfidfVectorizer(max_features=60_000, ngram_range=(1,3), stop_words=list(STOP_WORDS) if STOP_WORDS else 'english')
Xtr = vec.fit_transform(X_train_txt); Xte = vec.transform(X_test_txt)

cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
svm = LinearSVC(C=1.5, class_weight={i:w for i,w in enumerate(cw)}, random_state=SEED)
svm.fit(Xtr, y_train)
svm_pred = svm.predict(Xte)
svm_f1 = f1_score(y_test, svm_pred, average='macro')
svm_acc = accuracy_score(y_test, svm_pred)
summary.append(["TFIDF_SVM", svm_acc, svm_f1])

pickle.dump({"vect":vec,"clf":svm}, open(PROC_DIR/"TFIDF_SVM.pkl","wb"))
save_cm(y_test, svm_pred, "TF‑IDF + SVM", CM_DIR/"TFIDF_SVM_cm.png")

# ─── 7. SBERT + XGBoost -----------------------------------------------------
print("▶︎ SBERT + XGBoost (all-mpnet)")
bert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
Xtr_e = bert.encode(X_train_txt, batch_size=64, show_progress_bar=True)
Xte_e = bert.encode(X_test_txt,  batch_size=64, show_progress_bar=True)

xgb = XGBClassifier(n_estimators=800,max_depth=10,learning_rate=0.07,subsample=0.9,colsample_bytree=0.9,
                    objective='multi:softprob',eval_metric='mlogloss',random_state=SEED,n_jobs=-1)
xgb.fit(Xtr_e, y_train)
xgb_pred = xgb.predict(Xte_e)
xgb_f1 = f1_score(y_test, xgb_pred, average='macro')
xgb_acc = accuracy_score(y_test, xgb_pred)
summary.append(["SBERT_XGB", xgb_acc, xgb_f1])

pickle.dump({"embed":"all-mpnet-base-v2","clf":xgb}, open(PROC_DIR/"SBERT_XGB.pkl","wb"))
save_cm(y_test, xgb_pred, "SBERT + XGBoost", CM_DIR/"SBERT_XGB_cm.png")

# ─── 8. Özet & En İyi Model -------------------------------------------------
sum_df = pd.DataFrame(summary, columns=["model","accuracy","macro_f1"]).sort_values("macro_f1", ascending=False)
sum_df.to_csv(DATA_DIR/"mobile_security_models_summary.csv", index=False)

best_name = sum_df.iloc[0]['model']
if best_name == "TFIDF_SVM":
    shutil.copy(PROC_DIR/"TFIDF_SVM.pkl", PROC_DIR/"best_model.pkl")
else:
    shutil.copy(PROC_DIR/"SBERT_XGB.pkl", PROC_DIR/"best_model.pkl")

print("\n✅ Eğitim tamamlandı – Sonuçlar:")
print(sum_df)
print(f"Çıktılar → {PROC_DIR}")
