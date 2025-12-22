import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ==============================================================================
API_URL = "http://127.0.0.1:8000"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "processed" / "dados_dashboard_nivel1.csv"
MODEL_FILE = BASE_DIR / "models" / "rf_model.joblib"
FEATURES_FILE = BASE_DIR / "models" / "features_list.joblib"

st.set_page_config(page_title="Painel Estratégico - Nível 1", layout="wide")

# ==============================================================================
# FUNÇÕES DE CARREGAMENTO (CACHED)
# ==============================================================================
@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        return None
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def load_model_and_features():
    """
    Carrega o modelo e a lista de features para a análise de XAI.
    Usamos cache_resource pois o modelo é um objeto pesado e estático.
    """
    if not MODEL_FILE.exists():
        return None, None
    model = joblib.load(MODEL_FILE)
    features = joblib.load(FEATURES_FILE)
    return model, features

# ==============================================================================
# INTERFACE PRINCIPAL
# ==============================================================================
def main():
    st.title("🏛️ Painel Estratégico: Órgãos Superiores (Nível 1)")
    st.markdown("**Escopo:** Análise exclusiva de Ministérios, Autarquias e Fundações (Matriz).")
    
    # Carregar dados
    df = load_data()
    model, feature_list = load_model_and_features()
    
    if df is None:
        st.error("Dados processados não encontrados. Execute 'python -m src.training' primeiro.")
        return

    # --- SIDEBAR: FILTROS ---
    st.sidebar.header("Filtros")
    perfis = st.sidebar.multiselect(
        "Filtrar por Perfil de Gestão", 
        options=df['perfil_estrategico'].unique(),
        default=df['perfil_estrategico'].unique()
    )
    df_filtered = df[df['perfil_estrategico'].isin(perfis)]

    # --- KPIs DE TOPO ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Órgãos", len(df_filtered))
    col2.metric("Média Geral (Nível 1)", f"{df_filtered['media_geral'].mean():.1f}")
    
    top_orgao = df_filtered.loc[df_filtered['media_geral'].idxmax()]
    col3.metric("Benchmark (Melhor Clima)", top_orgao['nome_unidade'], f"{top_orgao['media_geral']:.1f}")

    st.divider()

    # ==========================================================================
    # ABAS DA APLICAÇÃO
    # ==========================================================================
    tab1, tab2, tab3 = st.tabs(["Radar de Clusters", "Simulador Executivo", "Raio-X (XAI)"])
    
    # --------------------------------------------------------------------------
    # ABA 1: RADAR (DIAGNÓSTICO)
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader("Mapeamento de Riscos na Alta Gestão")
        
        fig = px.scatter(
            df_filtered, 
            x="gestor_visao_clara", 
            y="media_geral", 
            color="perfil_estrategico",
            size="quantidade_resposta", 
            hover_data=['nome_unidade', 'id_unidade'],
            color_discrete_map={
                'Crítico (Alerta)': '#ef553b', 
                'Em Transição': '#fecb52', 
                'Alta Performance': '#00cc96'
            },
            title="Matriz: Visão Clara vs Satisfação Geral"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("Ver dados brutos"):
            st.dataframe(df_filtered[['nome_unidade', 'perfil_estrategico', 'media_geral']].sort_values('media_geral'))

    # --------------------------------------------------------------------------
    # ABA 2: SIMULADOR (PREDIÇÃO VIA API)
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("🔮 Simulador de Impacto")
        st.info("Ajuste as alavancas de liderança para projetar o novo clima organizacional.")
        
        col_input, col_result = st.columns([1, 2])
        
        with col_input:
            with st.form("simulacao_form"):
                gestor_coerencia = st.slider("Coerência (Walk the Talk)", 0.0, 100.0, float(df['gestor_coerencia_acao'].mean()))
                gestor_visao = st.slider("Clareza de Visão", 0.0, 100.0, float(df['gestor_visao_clara'].mean()))
                gestor_competencia = st.slider("Competência Técnica", 0.0, 100.0, float(df['gestor_competente'].mean()))
                promocao_merito = st.slider("Promoção por Mérito", 0.0, 100.0, float(df['promocao_merito'].mean()))
                
                submitted = st.form_submit_button("Calcular Impacto")
        
        with col_result:
            if submitted:
                # Prepara payload com defaults
                defaults = df.mean(numeric_only=True)
                payload = defaults.to_dict()
                payload.update({
                    'gestor_coerencia_acao': gestor_coerencia,
                    'gestor_visao_clara': gestor_visao,
                    'gestor_competente': gestor_competencia,
                    'promocao_merito': promocao_merito
                })
                
                try:
                    response = requests.post(f"{API_URL}/predict/simulacao", json=payload)
                    if response.status_code == 200:
                        nota_prevista = response.json()['nota_projetada']
                        
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = nota_prevista,
                            delta = {'reference': df['media_geral'].mean(), 'position': "top"},
                            title = {'text': "Média Geral Projetada"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#2b0955"},
                                'steps': [
                                    {'range': [0, 40], 'color': "#ef553b"},
                                    {'range': [40, 60], 'color': "#fecb52"},
                                    {'range': [60, 100], 'color': "#00cc96"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig_gauge)
                    else:
                        st.error("Erro na API.")
                except Exception as e:
                    st.error(f"Falha na conexão com API: {e}")

    # --------------------------------------------------------------------------
    # ABA 3: RAIO-X XAI (EXPLICABILIDADE)
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("🕵️‍♀️ Explicabilidade da IA (SHAP Waterfall)")
        st.markdown("""
        Entenda **exatamente** por que um órgão recebeu sua nota. Este gráfico decompõe a nota final 
        mostrando quanto cada fator contribuiu positiva (vermelho) ou negativamente (azul) em relação à média geral.
        """)

        if model is None:
            st.warning("Modelo não encontrado. Não é possível gerar explicações.")
        else:
            col_sel, col_graph = st.columns([1, 3])
            
            with col_sel:
                orgao_selecionado = st.selectbox(
                    "Selecione um Órgão para Auditar:", 
                    df_filtered['nome_unidade'].unique()
                )
                btn_explicar = st.button("Gerar Explicação")

            with col_graph:
                if btn_explicar:
                    with st.spinner("A IA está analisando os fatores de influência..."):
                        # 1. Encontrar os dados do órgão selecionado
                        row_data = df_filtered[df_filtered['nome_unidade'] == orgao_selecionado]
                        if not row_data.empty:
                            # Preparar X para o SHAP (apenas as features usadas no treino)
                            X_instance = row_data[feature_list]
                            
                            # 2. Calcular SHAP
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_instance)
                            
                            # 3. Plotar
                            # Criamos uma figura matplotlib explicitamente para passar ao streamlit
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # O objeto Explanation é necessário para o waterfall plot moderno
                            explanation = shap.Explanation(
                                values=shap_values[0], 
                                base_values=explainer.expected_value[0], 
                                data=X_instance.iloc[0], 
                                feature_names=feature_list
                            )
                            
                            shap.plots.waterfall(explanation, show=False)
                            st.pyplot(fig)
                            
                            st.caption(f"Análise baseada no modelo Random Forest (R²=0.92). "
                                       f"A nota base (E[f(x)]) é a média esperada, e as barras mostram os desvios.")

if __name__ == "__main__":
    main()