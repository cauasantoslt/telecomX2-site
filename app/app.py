# app.py

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import numpy as np

# Configurações da página
st.set_page_config(layout="wide")

# ====================================================================
# Constantes e Carregamento de Recursos
# ====================================================================

# Definindo a paleta de cores consistente
CHURN_COLORS = {'0': '#1f77b4', '1': '#ff7f0e'}  # Azul para 'Não Evadiu', Laranja para 'Evadiu'
PLOTLY_CMAP = 'RdBu_r'

# Carregar o DataFrame dos dados já processados para os gráficos
@st.cache_data
def load_data():
    """Carrega e pré-processa os dados para os gráficos."""
    try:
        # Caminho ajustado para a sua estrutura de pastas
        # CORREÇÃO: O arquivo agora está na mesma pasta que app.py, então removemos o caminho relativo.
        df = pd.read_csv('dados_tratados.csv')
        df = df.drop(columns=['customerID', 'Contas_Diarias'])
        
        # Correção: remove linhas com valores ausentes em 'Churn' e converte para string
        df = df.dropna(subset=['Churn'])
        df['Churn'] = df['Churn'].astype(str)
        
        # Separar a coluna Churn do restante do DataFrame para o one-hot encoding
        churn_column = df['Churn']
        df_to_encode = df.drop(columns=['Churn'])
        
        df_encoded = pd.get_dummies(df_to_encode, drop_first=True)
        
        # Adicionar a coluna Churn de volta ao DataFrame codificado
        df_encoded['Churn'] = churn_column.values

        return df, df_encoded
    except FileNotFoundError:
        st.error("Erro: O arquivo 'dados_tratados.csv' não foi encontrado. Certifique-se de que ele está no diretório correto.")
        return None, None

# Carregar o modelo, o scaler e as colunas (melhor prática)
@st.cache_resource
def load_model_resources():
    """Carrega os recursos do modelo salvos em arquivos .pkl."""
    try:
        model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, feature_columns
    except FileNotFoundError:
        st.error("Erro: Arquivos do modelo (.pkl) não foram encontrados. Por favor, treine o modelo e salve-o primeiro no seu notebook.")
        return None, None, None

df_raw, df_encoded = load_data()
model, scaler, feature_columns = load_model_resources()

# ====================================================================
# Layout da Aplicação Streamlit
# ====================================================================

# Título e cabeçalho
st.title('TelecomX - Análise de Evasão de Clientes (Churn)')
st.markdown("### Relatório Detalhado e Estratégias de Retenção")

# ====================================================================
# Seção 1: Resumo Executivo
# ====================================================================
st.header('1. Resumo Executivo')
st.markdown("""
Este relatório apresenta os resultados de um projeto de Machine Learning com o objetivo de prever a evasão de clientes (Churn) e identificar os fatores mais relevantes que levam a essa evasão. O modelo **Random Forest** foi selecionado por seu desempenho superior, alcançando um Recall de 91%, indicando uma alta capacidade de identificar clientes em risco de evasão.
""")

# ====================================================================
# Seção 2: Fatores Chave que Influenciam a Evasão
# ====================================================================
st.header('2. Fatores Chave de Evasão')
st.markdown("""
A análise de importância de variáveis do modelo Random Forest revelou os seguintes insights:

- **Tempo de Contrato (`customer_tenure`):** É o fator mais importante. Clientes novos têm maior propensão a evadir.
- **Tipo de Contrato (`account_Contract`):** Clientes com contratos `mês a mês` são mais propensos a sair.
- **Serviços Adicionais:** A falta de `OnlineSecurity` e `TechSupport` aumenta o risco de churn.
""")

if model and df_encoded is not None:
    # Recriar o DataFrame de importância de features para o gráfico
    # Certifique-se de usar as colunas corretas após o one-hot encoding
    X = df_encoded.drop('Churn', axis=1)
    importance_df = pd.DataFrame({
        'Importância': model.feature_importances_,
        'Variável': X.columns
    }).sort_values(by='Importância', ascending=False).head(10)

    fig_imp = px.bar(importance_df,
                     x='Importância',
                     y='Variável',
                     title='Top 10 Variáveis mais Importantes (Random Forest)',
                     color_discrete_sequence=px.colors.sequential.Plotly3)
    st.plotly_chart(fig_imp)

# ====================================================================
# Seção 3: Visualizações Detalhadas
# ====================================================================
st.header('3. Visualizações Detalhadas')
st.write('Explore a seguir as principais distribuições de variáveis em relação à evasão de clientes.')

if df_raw is not None:
    # Boxplot para Tempo de Contrato
    fig_tenure = px.box(df_raw,
                        x='Churn',
                        y='customer_tenure',
                        color='Churn',
                        color_discrete_map=CHURN_COLORS,
                        title='Distribuição de Tempo de Contrato por Evasão')
    st.plotly_chart(fig_tenure)

    # Boxplot para Gasto Total
    fig_total_charges = px.box(df_raw,
                               x='Churn',
                               y='account_Charges.Total',
                               color='Churn',
                               color_discrete_map=CHURN_COLORS,
                               title='Distribuição de Gasto Total por Evasão')
    st.plotly_chart(fig_total_charges)

    # Matriz de Correlação
    st.subheader('Matriz de Correlação das Variáveis')
    corr_matrix = df_encoded.corr()
    fig_corr = px.imshow(corr_matrix,
                         labels=dict(x="Variáveis", y="Variáveis", color="Correlação"),
                         x=corr_matrix.columns,
                         y=corr_matrix.columns,
                         color_continuous_scale=PLOTLY_CMAP,
                         title='Matriz de Correlação Interativa')
    st.plotly_chart(fig_corr)

# ====================================================================
# Seção 4: Estratégias de Retenção
# ====================================================================
st.header('4. Estratégias de Retenção Propostas')
st.markdown("""
Com base nos insights obtidos, recomendamos as seguintes ações:
- **Programa de Engajamento para Novos Clientes:** Desenvolver um programa de onboarding focado nos primeiros meses de contrato.
- **Incentivos para Contratos de Longo Prazo:** Oferecer descontos e benefícios para clientes que migrarem de contratos mensais para anuais.
- **Pacotes de Serviços Agregados:** Criar pacotes que combinem internet de alta velocidade com serviços de segurança e suporte técnico.
- **Ações Proativas de Atendimento:** Utilizar o modelo de previsão para identificar clientes em risco e abordá-los proativamente.
""")

# ====================================================================
# Seção 5: Previsão de Novo Cliente (Funcionalidade Extra)
# ====================================================================
st.header('5. Previsão de Churn para um Novo Cliente')
st.write('Insira os dados de um cliente para prever sua probabilidade de evasão (Churn).')

if model and scaler and feature_columns:
    colunas_numericas = ['customer_tenure', 'account_Charges.Monthly', 'account_Charges.Total']
    
    with st.form("churn_prediction_form"):
        # Interface de entrada de dados (simplificada para demonstração)
        st.subheader("Informações do Cliente")
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider('Tempo de Contrato (meses)', min_value=1, max_value=72, value=12)
            monthly_charges = st.slider('Gasto Mensal', min_value=18.0, max_value=118.0, value=70.0)
            contract = st.selectbox('Tipo de Contrato', ['month-to-month', 'one year', 'two year'])
        with col2:
            total_charges = st.number_input('Gasto Total', min_value=0.0, value=1000.0)
            internet_service = st.selectbox('Serviço de Internet', ['dsl', 'fiber optic', 'no internet service'])
            online_security = st.selectbox('Segurança Online (1=Sim, 0=Não)', [0, 1])
            tech_support = st.selectbox('Suporte Técnico (1=Sim, 0=Não)', [0, 1])

        # Botão de submissão
        submitted = st.form_submit_button("Prever Churn")

        if submitted:
            # Criar um DataFrame com os dados de entrada
            input_data = pd.DataFrame([{
                'customer_tenure': tenure,
                'account_Charges.Monthly': monthly_charges,
                'account_Charges.Total': total_charges,
                'internet_OnlineSecurity_Yes': online_security,
                'internet_TechSupport_Yes': tech_support,
                'account_Contract_One year': 1 if contract == 'one year' else 0,
                'account_Contract_Two year': 1 if contract == 'two year' else 0,
                'internet_InternetService_Fiber optic': 1 if internet_service == 'fiber optic' else 0,
                'internet_InternetService_No internet service': 1 if internet_service == 'no internet service' else 0,
            }])

            # Adicionar colunas faltantes com valor 0 para garantir consistência
            for col in feature_columns:
                if col not in input_data.columns:
                    input_data[col] = 0

            # Padronizar os dados numéricos do novo cliente
            input_data[colunas_numericas] = scaler.transform(input_data[colunas_numericas])

            # Fazer a previsão
            churn_prediction = model.predict(input_data[feature_columns])
            churn_prob = model.predict_proba(input_data[feature_columns])[0][1]

            if churn_prediction[0] == 1:
                st.error(f'Atenção: Este cliente tem alta probabilidade de evasão! Probabilidade: {churn_prob:.2f}')
            else:
                st.success(f'Este cliente tem baixa probabilidade de evasão. Probabilidade: {churn_prob:.2f}')

else:
    st.warning("Não foi possível carregar os recursos necessários. Verifique se os arquivos `.pkl` existem.")
