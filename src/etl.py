# src/etl.py
import pandas as pd
import re
from . import config

def parse_subclasse(text):
    """Extrai ID e Nome da string de subclasse."""
    match = re.match(r'([\d\.]+)\s*-\s*(.+)', str(text))
    if match:
        return match.group(1), match.group(2)
    return None, text

def load_and_clean_data(filepath=config.DATA_PATH):
    """Pipeline de Ingestão e Limpeza."""
    df_raw = pd.read_csv(filepath)

    # Filtrar Escopo: Órgão + Empresa
    df = df_raw[
        (df_raw['classe'] == 'Órgão') & 
        (df_raw['area_empresa'] == 'Empresa')
    ].copy()

    # Tratamento de Hierarquia
    parsed = df['subclasse'].apply(parse_subclasse)
    df['id_unidade'] = [x[0] for x in parsed]
    df['nome_unidade'] = [x[1] for x in parsed]
    df['nivel_hierarquico'] = df['id_unidade'].apply(lambda x: str(x).count('.') + 1 if x else 0)

    # Seleção de Colunas
    cols_meta = ['id_unidade', 'nome_unidade', 'nivel_hierarquico', 'quantidade_resposta', 'media_geral']
    
    # Garantir que as features existem
    available_features = [f for f in config.FEATURES_LIDERANCA if f in df.columns]
    
    df_clean = df[cols_meta + available_features].dropna(subset=['media_geral'])
    
    return df_clean, available_features