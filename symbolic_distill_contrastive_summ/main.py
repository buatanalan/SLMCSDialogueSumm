from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 1. Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="Transformer Summarization API",
    description="API untuk serving model NLP secara eksplisit (tanpa pipeline)",
    version="1.0.0"
)

# 2. Muat model saat aplikasi dijalankan (Global state)
MODEL_PATH = "method4_outputs/bart_margin_part/outputs/checkpoint-570"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    print(f"Memuat model dan tokenizer dari {MODEL_PATH} ke {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    nlp_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
    # Set model ke mode evaluasi (penting untuk dropout/batchnorm)
    nlp_model.eval() 
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    tokenizer = None
    nlp_model = None

# 3. Definisikan skema input (Payload) menggunakan Pydantic
class InferenceRequest(BaseModel):
    text: str
    max_length: int = 150  # Batas maksimum token ringkasan
    min_length: int = 30   # Batas minimum token ringkasan

# 4. Buat Endpoint untuk inferensi
@app.post("/predict")
async def predict(request: InferenceRequest):
    if nlp_model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model belum siap atau gagal dimuat.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input teks tidak boleh kosong.")

    try:
        # A. Tokenisasi: Mengubah teks menjadi tensor
        inputs = tokenizer(
            request.text, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True
        ).to(device)
        
        # B. Generate: Memprediksi token ringkasan
        # Menggunakan torch.no_grad() untuk menghemat memori karena tidak ada backpropagation
        with torch.no_grad():
            output_ids = nlp_model.generate(
                **inputs,
                max_length=request.max_length,
                min_length=request.min_length,
                num_beams=4,             # Beam search untuk hasil yang lebih masuk akal
                early_stopping=True      # Berhenti ketika ringkasan selesai
            )
        
        # C. Dekode: Mengubah output tensor kembali menjadi teks
        summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Format output
        return {
            "input_text": request.text,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat inferensi: {str(e)}")

# Endpoint untuk Health Check
@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": nlp_model is not None}