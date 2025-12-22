# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Configuração de Caminhos (Ajuste conforme onde você roda o uvicorn)
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = FastAPI(title="API Clima Organizacional", version="1.0")

# Carregar Modelos na Inicialização
try:
    rf_model = joblib.load(MODELS_DIR / "rf_model.joblib")
    features_list = joblib.load(MODELS_DIR / "features_list.joblib")
except Exception as e:
    print(f"Erro ao carregar modelos: {e}")

# Schema de Validação (O que a API espera receber)
class SimulacaoInput(BaseModel):
    # Cria campos dinamicamente ou fixos
    gestor_competente: float
    gestor_coerencia_acao: float
    gestor_cumprimento_promessa: float
    gestor_visao_clara: float
    gestor_honestidade_etica: float
    gestor_sem_favoritismo: float
    gestor_envolvimento_decisao: float
    gestor_resposta_direta: float
    oferta_desenvolvimento_profissional: float
    promocao_merito: float
    ambiente_fisicamente_seguro: float
    ambiente_psicologicamente_saudavel: float

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": rf_model is not None}

@app.post("/predict/simulacao")
def predict_score(data: SimulacaoInput):
    """Recebe notas de liderança e retorna a Média Geral prevista."""
    try:
        # Converter input para DataFrame na ordem correta das features
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Garantir ordem das colunas
        input_df = input_df[features_list]
        
        # Previsão
        prediction = rf_model.predict(input_df)[0]
        
        return {
            "nota_projetada": round(float(prediction), 2),
            "input_recebido": input_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))