# src/training.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from . import etl, config

def train_pipeline():
    print("--- Iniciando Pipeline de Treinamento (Escopo: Nível Hierárquico 1) ---")
    
    # 1. Carregar Dados
    df_full, features = etl.load_and_clean_data()
    
    # === FILTRO DE ESCOPO: APENAS NÍVEL 1 ===
    # Mantém apenas Ministérios, Autarquias Matriz e Órgãos Superiores
    df = df_full[df_full['nivel_hierarquico'] == 1].copy()
    
    print(f"Dados filtrados. Total de Órgãos Superiores para análise: {len(df)}")
    
    if len(df) < 10:
        print("AVISO: Poucos dados para treinamento robusto neste nível.")

    # Preparar X e y
    X = df[features].dropna()
    y = df.loc[X.index, config.TARGET]

    # 2. Treinar Clusterização (Radar Estratégico)
    print("Treinando Clusterização Estratégica...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reduzimos para 3 clusters que agora representam perfis de Alta Gestão
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)
    
    # 3. Treinar Simulador (Regressão)
    print("Treinando Simulador de Alta Gestão...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 4. Salvar Artefatos
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")
    joblib.dump(kmeans, config.MODELS_DIR / "kmeans_model.joblib")
    joblib.dump(rf, config.MODELS_DIR / "rf_model.joblib")
    joblib.dump(features, config.MODELS_DIR / "features_list.joblib")
    
    # Salvar dados processados para o Dashboard (Apenas Nível 1)
    df_dashboard = df.loc[X.index].copy()
    df_dashboard['cluster'] = kmeans.labels_
    
    # Mapeamento automático dos nomes dos clusters baseado na média geral
    cluster_means = df_dashboard.groupby('cluster')['media_geral'].mean().sort_values()
    # Ex: 0 -> Baixo, 1 -> Médio, 2 -> Alto (a ordem do k-means é aleatória, isso corrige)
    rank_map = {old_label: rank for rank, old_label in enumerate(cluster_means.index)}
    
    # Labels semânticas para o Dashboard
    labels_map = {0: 'Crítico (Alerta)', 1: 'Em Transição', 2: 'Alta Performance'}
    df_dashboard['cluster_rank'] = df_dashboard['cluster'].map(rank_map)
    df_dashboard['perfil_estrategico'] = df_dashboard['cluster_rank'].map(labels_map)
    
    # Salvar CSV final
    output_path = config.DATA_PATH.parent.parent / "processed" / "dados_dashboard_nivel1.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_dashboard.to_csv(output_path, index=False)

    print(f"Pipeline concluído. Dados salvos em: {output_path}")

if __name__ == "__main__":
    train_pipeline()