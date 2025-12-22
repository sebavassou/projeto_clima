# src/config.py
from pathlib import Path

# Caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "br_me_clima_organizacional_microdados.csv"
MODELS_DIR = BASE_DIR / "models"

# Features utilizadas nos modelos
FEATURES_LIDERANCA = [
    'gestor_competente', 'gestor_coerencia_acao', 'gestor_cumprimento_promessa',
    'gestor_visao_clara', 'gestor_honestidade_etica', 'gestor_sem_favoritismo',
    'gestor_envolvimento_decisao', 'gestor_resposta_direta', 
    'oferta_desenvolvimento_profissional', 'promocao_merito',
    'ambiente_fisicamente_seguro', 'ambiente_psicologicamente_saudavel'
]

TARGET = 'media_geral'