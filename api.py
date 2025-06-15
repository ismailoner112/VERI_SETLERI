# app/api.py
import json, joblib, pathlib, io
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
from skimage.feature import hog

# -----------------------------------------------------------------------------
# 1) MODEL DİZİNİ
ROOT = pathlib.Path(__file__).parent / "unified_models"
INDEX = json.load(open(ROOT / "model_index.json", encoding="utf-8"))

LABEL_DIR = ROOT / "labels"
CACHE: dict[str, Any] = {}


def load(ds: str):
    if ds not in INDEX:
        raise HTTPException(404, f"\'{ds}\' listede yok.")
    if ds not in CACHE:
        # Ensure INDEX[ds] is a dictionary and has a 'file' key
        if not isinstance(INDEX.get(ds), dict) or "file" not in INDEX[ds]:
             raise HTTPException(500, f"\'{ds}\' modeli için model yolu bulunamadı veya format hatalı.")

        file_name = INDEX[ds]["file"]
        file = ROOT / file_name
        if not file.exists():
             raise HTTPException(404, f"Model dosyası bulunamadı: {file}")
        try:
            if file.suffix == ".pkl":
                # Check if it's a text model that might need SentenceTransformer
                model_data = joblib.load(file)
                if isinstance(model_data, dict) and ("embed" in model_data or "vect" in model_data):
                     # If embed key exists and is a string (model name), load SentenceTransformer
                     if "embed" in model_data and isinstance(model_data["embed"], str):
                         # Load SentenceTransformer only when needed for prediction
                         try:
                             from sentence_transformers import SentenceTransformer
                             if "_sbert_model" not in CACHE:
                                  CACHE["_sbert_model"] = SentenceTransformer(model_data["embed"])
                             sbert_model = CACHE["_sbert_model"]
                         except ImportError:
                              raise HTTPException(500, detail="'sentence_transformers' kütüphanesi yüklü değil. Metin analizi için yükleyin.")
                         except Exception as e:
                              raise HTTPException(500, detail=f"SentenceTransformer modeli yüklenirken hata oluştu: {e}")

                     # Store the loaded model data
                     CACHE[ds] = model_data
                elif isinstance(model_data, dict) and "clf" in model_data:
                     # Assume it's an image model if it has clf but not text components
                      CACHE[ds] = model_data
                else:
                     # For models that don't fit the expected dict structure
                     CACHE[ds] = model_data # Cache it anyway, might be a simple model

            elif file.suffix == ".h5":
                import tensorflow as tf
                import os
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
                CACHE[ds] = tf.keras.models.load_model(file)
            else:
                raise HTTPException(500, "Desteklenmeyen model formatı")
        except Exception as e:
             raise HTTPException(500, detail=f"Model yüklenirken hata oluştu: {e}")

    return CACHE[ds]


def label_name(ds: str, idx: int) -> str:
    f = LABEL_DIR / f"{ds}.json"
    if f.exists():
        try:
            labels = json.load(open(f, encoding="utf-8"))
            return labels[int(idx)]
        except IndexError:
            # If index is out of bounds, return the index as string and print a warning
            print(f"Uyarı: \'{ds}\' etiket dosyası için indeks sınırlar dışında: {idx}")
            return str(idx)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Uyarı: {ds} için etiket dosyası okunurken hata: {e}")
            pass # Should not reach here if IndexError is caught first, but keep as fallback
    return str(idx)



# -----------------------------------------------------------------------------
# 2) FASTAPI
app = FastAPI(title="Güvenlik Analiz Sistemi", version="1.0")

# Frontend klasörünü statik dosyalar olarak sun
app.mount("/frontend_assets", StaticFiles(directory="frontend"), name="frontend_assets")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ana sayfa - index.html içeriğini döndür
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


class TextIn(BaseModel):
    text: str
    dataset: str


@app.get("/datasets")
def list_ds() -> List[Dict[str, Any]]:
    dataset_info = []
    for ds_name in INDEX.keys():
        try:
            # Get supported types directly from the INDEX dictionary
            supported_types = INDEX[ds_name].get("types", []) if isinstance(INDEX.get(ds_name), dict) else []

   

            dataset_info.append({"name": ds_name, "types": supported_types})
        except Exception as e:
             print(f"Uyarı: {ds_name} modeli bilgisi işlenirken beklenmeyen hata: {e}")
             # Still include dataset with error info if its info can't be processed
             dataset_info.append({"name": ds_name, "types": [], "error": str(e)})

    # Debugging print to see what dataset_info contains
    # print(f"DEBUG: dataset_info: {dataset_info}")

    return dataset_info


@app.post("/predict_text")
def predict_text(inp: TextIn):
    mdl = load(inp.dataset)

    # Check if the loaded model has the necessary components for text analysis
    is_text_model = False
    if isinstance(mdl, dict):
        if "vect" in mdl:
            is_text_model = True
        elif "embed" in mdl and isinstance(mdl["embed"], str):
            # Assume it's a text model if it has an 'embed' key with a string value (SentenceTransformer name)
            is_text_model = True

    if not is_text_model:
         # Use the detail from the HTTPException raised here
        raise HTTPException(500, detail=f"Seçilen \'{inp.dataset}\' veri setinin modeli metin analizi için uygun bileşenlere sahip değil veya tanımsız formatta.")

    if "vect" in mdl:
        X = mdl["vect"].transform([inp.text])
        pred_idx = int(mdl["clf"].predict(X)[0])
    elif "embed" in mdl:
        # Ensure sentence_transformers is installed if using embed
        try:
             
             if isinstance(mdl["embed"], str):
                
                 try:
                     from sentence_transformers import SentenceTransformer
                     if "_sbert_model" not in CACHE:
                          CACHE["_sbert_model"] = SentenceTransformer(mdl["embed"])
                     sbert_model = CACHE["_sbert_model"]
                 except ImportError:
                      raise HTTPException(500, detail="'sentence_transformers' kütüphanesi yüklü değil. Metin analizi için yükleyin.")
                 except Exception as e:
                      raise HTTPException(500, detail=f"SentenceTransformer modeli yüklenirken hata oluştu: {e}")

             elif hasattr(mdl["embed"], 'encode'):
                 # Assuming the SentenceTransformer object is directly in mdl["embed"]
                 sbert_model = mdl["embed"]
             else:
                 raise ValueError("Desteklenmeyen embed modeli formatı")

             emb = sbert_model.encode([inp.text])
             pred_idx = int(mdl["clf"].predict(emb)[0])
        except Exception as e:
             raise HTTPException(500, detail=f"Metin gömme veya tahmin sırasında hata oluştu: {e}")

    return {
        "dataset": inp.dataset,
        "label": label_name(inp.dataset, pred_idx),
        "code": pred_idx
    }


def extract_features(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img)
    # Ensure image data is uint8 before HOG for consistent behavior
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    hist, _ = np.histogram(arr, bins=64, range=(0, 255))
    hist = hist.astype(float) / (hist.sum() + 1e-9)
    
    try:
        # HOG may fail on very small or problematic images
        hog_vec = hog(arr, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), feature_vector=True)
    except Exception as e:
        print(f"Uyarı: HOG özelliği çıkarılırken hata: {e}")
        # Return zero vector or handle as error
        hog_vec = np.zeros((get_hog_feature_vector_size((128,128), (16,16), (2,2)))) # Assuming default size

    return np.hstack([hist, hog_vec])

# Helper to calculate expected HOG feature size (approximation)
def get_hog_feature_vector_size(image_size, pixels_per_cell, cells_per_block):
    img_height, img_width = image_size
    cell_height, cell_width = pixels_per_cell
    block_height, block_width = cells_per_block

    # Number of cells in each direction
    n_cells_y = int(img_height / cell_height)
    n_cells_x = int(img_width / cell_width)

    # Number of blocks in each direction
    n_blocks_y = n_cells_y - block_height + 1
    n_blocks_x = n_cells_x - block_width + 1

    # Number of orientations (common default is 9)
    n_orientations = 9 # Default in scikit-image hog

    # Features per block = cells_per_block * cells_per_block * orientations
    features_per_block = block_height * block_width * n_orientations

    # Total features = number of blocks * features_per_block
    total_features = n_blocks_y * n_blocks_x * features_per_block
    
    # This is an approximation, actual size might vary slightly based on library implementation details
    return total_features


@app.post("/predict_image")
async def predict_image(dataset: str, file: UploadFile = File(...)):
    mdl = load(dataset)

    # Check if the model supports image analysis based on INDEX and loaded model structure
    dataset_index_info = INDEX.get(dataset, {})
    supports_image_in_index = "image" in dataset_index_info.get("types", [])

    # Also check the structure of the loaded model as a fallback/validation
    supports_image_in_model = isinstance(mdl, dict) and "clf" in mdl

    if not (supports_image_in_index or supports_image_in_model):
         # Use the detail from the HTTPException raised here
        raise HTTPException(500, detail="Seçilen veri seti görsel analizi için uygun değil.")

    # Görseli yükle ve gri tonlamaya çevir
    try:
        pil_img = Image.open(io.BytesIO(await file.read())).convert("L")
    except Exception as e:
        raise HTTPException(400, detail=f"Görsel dosyası okunurken hata oluştu: {e}")

    # Modelin beklentisine göre görseli yeniden boyutlandır ve işle
    # Bu kısım hala varsayımlara dayanıyor. Model index dosyasına boyut/işleme bilgisi eklemek en iyisi.
    expected_features = None # We don't know the expected features generically

    pil_img_processed = None
    vec = None

    # Attempt to infer expected processing based on model structure or dataset name
    if dataset == "Mobile_Security_Dataset":
        # Mobile_Security_Dataset için 200x300 boyut bekleniyor (60000 özellik varsayımı)
        pil_img_processed = pil_img.resize((300, 200))
        vec = np.array(pil_img_processed).flatten() / 255.0
    elif "scaler" in mdl:
        # Model extract_features ve scaler bekliyorsa (varsayılan 128x128 boyut varsayımı)
        pil_img_processed = pil_img.resize((128, 128))
        vec = extract_features(pil_img_processed)
        if mdl["scaler"] is not None:
            try:
                vec = mdl["scaler"].transform([vec])[0]
            except ValueError as e:
                 raise HTTPException(500, detail=f"Model beklentisi ile görsel özellik sayısı uyuşmuyor: {e}")
    else:
        # Varsayılan olarak 128x128 boyut ve flatten (16384 özellik)
        pil_img_processed = pil_img.resize((128, 128))
        vec = np.array(pil_img_processed).flatten() / 255.0
        # If model expects 60000 and we flattened 128x128, this will fail later
        # A better approach is needed to know the exact expected input shape/features per model

    if vec is None:
         raise HTTPException(500, detail="Görsel işleme adımı tamamlanamadı.")

    # Modelin beklediği giriş boyutuyla uyuşup uyuşmadığını kontrol etmek daha iyi olurdu
    # try:
    #     # Bu kontrol modelin predict metoduna göndermeden önce yapılmalıydı
    #     # Ancak model objesinin beklediği boyutu dinamik olarak öğrenmek zor olabilir
    #     pass 
    # except Exception as e:
    #      raise HTTPException(500, detail=f"Görsel boyutu/özelliği modelle uyumsuz: {e}")


    try:
        # Tahmin yap
        # Modeller tek örnek bekleyebilir, bu yüzden [vec] olarak gönderiyoruz
        pred_idx = int(mdl["clf"].predict([vec])[0])
    except ValueError as e:
         # LinearSVC gibi modeller burada boyut hatası fırlatabilir
         raise HTTPException(500, detail=f"Model tahmin girişi ile ilgili hata: {e}. Lütfen doğru veri setini ve görsel formatını seçtiğinizden emin olun.")
    except Exception as e:
         raise HTTPException(500, detail=f"Tahmin sırasında beklenmeyen bir hata oluştu: {e}")

    return {
        "dataset": dataset,
        "label": label_name(dataset, pred_idx),
        "code": pred_idx
    }
