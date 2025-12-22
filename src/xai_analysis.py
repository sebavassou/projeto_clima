import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# Importar configurações e ETL do nosso projeto
# (Assumindo que você está na raiz do projeto criado anteriormente)
from src import etl, config

def gerar_analise_explicabilidade():
    print("--- Iniciando Análise de XAI (LIME & SHAP) ---")
    
    # 1. Carregar Dados e Modelo
    # Se você já rodou o training.py, carregamos o modelo treinado.
    # Caso contrário, retreinamos aqui rapidamente.
    try:
        model = joblib.load(config.MODELS_DIR / "rf_model.joblib")
        features_list = joblib.load(config.MODELS_DIR / "features_list.joblib")
        print("Modelo carregado com sucesso.")
    except:
        print("Modelo não encontrado. Por favor, execute src/training.py primeiro.")
        return

    # Carregar dados (Nível 1 apenas, conforme definido anteriormente)
    df_full, _ = etl.load_and_clean_data()
    df = df_full[df_full['nivel_hierarquico'] == 1].copy()
    X = df[features_list].dropna()
    
    # Resetar index para garantir alinhamento
    X = X.reset_index(drop=True)
    df_dashboard = df.loc[X.index].reset_index(drop=True) # Metadados para identificar os órgãos

    # ==========================================================================
    # A. ANÁLISE SHAP (Global e Local)
    # ==========================================================================
    print("\nCalculando SHAP Values (Isso pode levar alguns segundos)...")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(X)

    # 1. Plot Global: Summary Plot (O mais importante)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Impacto Global: O que impulsiona (ou derruba) a nota?", fontsize=16)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png')
    print("-> Gráfico 'shap_summary_plot.png' salvo.")
    print("   (Interpretação: Pontos vermelhos à direita aumentam a nota. Azuis à esquerda diminuem.)")

    # 2. Plot Local: Waterfall (Diagnóstico de um Órgão Crítico)
    # Vamos pegar o pior órgão (menor média geral)
    idx_critico = df_dashboard['media_geral'].idxmin()
    nome_critico = df_dashboard.iloc[idx_critico]['nome_unidade']
    
    print(f"\nGerando Waterfall Plot para o Órgão Crítico: {nome_critico}")
    
    plt.figure(figsize=(10, 6))
    # SHAP waterfall plot requires an Explanation object in newer versions
    shap_explanation = shap.Explanation(
        values=shap_values[idx_critico], 
        base_values=explainer_shap.expected_value, 
        data=X.iloc[idx_critico], 
        feature_names=features_list
    )
    shap.plots.waterfall(shap_explanation, show=False)
    plt.title(f"Por que o órgão '{nome_critico}' teve essa nota?", fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_waterfall_critico.png')
    print("-> Gráfico 'shap_waterfall_critico.png' salvo.")

    # ==========================================================================
    # B. ANÁLISE LIME (Simulação Local)
    # ==========================================================================
    print("\nIniciando Análise LIME...")
    
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X),
        feature_names=features_list,
        class_names=['media_geral'],
        mode='regression'
    )

    # Vamos explicar um órgão "Mediano" (perto da média) para ver como ele pode subir
    # Encontrar órgão com nota próxima a 50
    idx_medio = (df_dashboard['media_geral'] - 50).abs().idxmin()
    nome_medio = df_dashboard.iloc[idx_medio]['nome_unidade']
    data_row = X.iloc[idx_medio]

    print(f"Explicando Órgão Mediano: {nome_medio} (Nota Real: {df_dashboard.iloc[idx_medio]['media_geral']:.1f})")
    
    exp = explainer_lime.explain_instance(
        data_row=data_row, 
        predict_fn=model.predict
    )
    
    # Salvar a explicação como HTML interativo
    with open('lime_explanation.html', 'w', encoding='utf-8') as f:
        f.write(exp.as_html())
    print("-> Arquivo 'lime_explanation.html' salvo. Abra no navegador.")
    
    # Mostrar regras extraídas pelo LIME no console
    print("\nRegras LIME (Impacto Local):")
    for feature_rule, weight in exp.as_list():
        impacto = "AUMENTA" if weight > 0 else "DIMINUI"
        print(f" - {feature_rule}: {impacto} a nota em {abs(weight):.2f} pontos")

if __name__ == "__main__":
    gerar_analise_explicabilidade()