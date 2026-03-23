# projeto_clima
# 🏛️ Governança Algorítmica e People Analytics Prescritivo no Setor Público

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-1.26.0-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-F7931E)
![SHAP](https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-8A2BE2)
![License](https://img.shields.io/badge/License-MIT-green)

Este repositório contém o código-fonte integral do artefato tecnológico desenvolvido como Trabalho de Conclusão de Curso (TCC) para o **MBA em Ciência de Dados e Inteligência Artificial da Escola Nacional de Administração Pública (Enap)**.

O projeto aplica a metodologia **Design Science Research (DSR)** para elevar a gestão de pessoas no setor público do nível descritivo para o **prescritivo**. Através de algoritmos de *Machine Learning* e Inteligência Artificial Explicável (XAI), o sistema diagnostica o clima organizacional e permite a simulação interativa do impacto de políticas de capacitação de lideranças.

---

## ✨ Funcionalidades do Artefato

1. **Diagnóstico Situacional (K-Means):** Triagem algorítmica não supervisionada que segmenta as unidades governamentais (Nível 1) em perfis de risco estratégico ("Crítico", "Em Transição" e "Alta Performance").
2. **Simulador Prescritivo (Random Forest):** Modelo supervisionado de *Ensemble Learning* capaz de projetar com alta aderência (R² = 0.84) a nota de clima organizacional com base nas variações de competências da liderança.
3. **Explicabilidade Híbrida (SHAP e LIME):** Aplicação do framework de IA Socialmente Responsável (SRAI) para garantir transparência nas predições, revelando os impulsionadores globais e locais da satisfação dos servidores.
4. **Governança Algorítmica (API REST):** Interoperabilidade garantida via FastAPI, com validação estrita de contratos de dados via Pydantic.
5. **Dashboard Interativo (Streamlit):** Interface Humano-Computador para uso direto por formuladores de políticas públicas e alta gestão.

---

## 📂 Arquitetura do Projeto

Para garantir a reprodutibilidade e a governança dos dados (em conformidade com a LGPD), a arquitetura separa estritamente o código da camada de dados.

```text
├── api/
│   └── main.py                 # Backend: API RESTful desenvolvida em FastAPI
├── app/
│   └── app.py                  # Frontend: Interface de simulação em Streamlit
├── data/                       # (Ignorado no Git por segurança)
│   ├── raw/                    # Microdados originais (.csv)
│   └── processed/              # Matrizes agregadas para o Dashboard
├── models/                     # Artefatos serializados (.joblib gerados no treino)
├── src/
│   ├── config.py               # Variáveis de ambiente e definição de features
│   ├── etl.py                  # Pipeline de Extração, Transformação e Carga
│   ├── training.py             # Treinamento do K-Means e Random Forest
│   └── xai_analysis.py         # Geração de interpretações SHAP e LIME
├── .gitignore
├── README.md
└── requirements.txt            # Dependências do projeto

🚀 Como Reproduzir o Ambiente
1. Pré-requisitos
Certifique-se de ter o Python 3.9+ instalado em sua máquina. Recomenda-se o uso de um ambiente virtual.
# Clone o repositório
git clone https://github.com/sebavassou/projeto_clima.git
cd projeto_clima

# Crie e ative o ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

2. Governança de Dados (Setup Local)
O arquivo bruto de microdados governamentais (br_me_clima_organizacional_microdados.csv) não está versionado neste repositório. Para executar o pipeline:

Crie a pasta data/raw/ na raiz do projeto.

Insira o arquivo CSV original dentro desta pasta.

3. Executando o Pipeline de Treinamento
Para processar os dados, realizar a engenharia de features e treinar os modelos matemáticos:

# Na raiz do projeto, execute o script de treinamento:
python -m src.training

Este comando populará a pasta models/ com os artefatos .joblib e a pasta data/processed/ com os arquivos sumarizados.

4. Iniciando a API (Backend)
O motor preditivo roda como um microsserviço independente. Em um terminal, inicie o servidor Uvicorn:

uvicorn api.main:app --reload

A API estará disponível em http://127.0.0.1:8000. Você pode testar os endpoints na documentação interativa em http://127.0.0.1:8000/docs.

5. Iniciando o Dashboard (Frontend)
Abra um novo terminal (mantendo a API rodando no anterior), ative o ambiente virtual e inicie a interface gerencial:

streamlit run app/app.py

👨‍💻 Autor
Sebastien Pierre Daniel Vassou é Coordenador Substituto de Administrção de Pessoas do Instituto Nacional de Traumatologia e Ortopedia, Técnologo em Big Data e Inteligência Analítica e  Pós Graduando (MBA) em Ciência de Dados e Inteligência Artificial na Escola Nacional de Administração Pública (Enap).

Este projeto é acadêmico e de código aberto (MIT License).
