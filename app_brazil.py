# app_brazil.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import os
import requests
import json
from dotenv import load_dotenv

# --- Initial Setup & Configuration ---
load_dotenv()

st.set_page_config(
    page_title="GS: Focos de Incêndio no Brasil",
    page_icon="🇧🇷🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hugging Face Inference API Configuration
HF_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

DATA_FILE_PATH = 'all_dashboard-fires-month-02-06-2025-22_23_57.csv'
MODEL_ARTIFACTS_PATHS = {
    "model": 'brazil_fire_focus_model.joblib',
    "classes": 'unique_classes.joblib',
    "ufs": 'unique_ufs.joblib',
    "features": 'model_feature_columns.joblib'
}

# --- Helper Functions & Data Loading ---
@st.cache_resource
def load_sklearn_model_assets(paths):
    try:
        model = joblib.load(paths["model"])
        unique_classes_val = joblib.load(paths["classes"])
        unique_ufs_val = joblib.load(paths["ufs"])
        model_feature_columns_val = joblib.load(paths["features"])
        return model, unique_classes_val, unique_ufs_val, model_feature_columns_val
    except FileNotFoundError:
        return None, None, None, None

@st.cache_data
def load_fire_data(file_path):
    try:
        df_raw = pd.read_csv(file_path, sep=';', encoding='utf-8-sig')
        df_raw.columns = df_raw.columns.str.strip()
        if df_raw.columns[0].startswith('\ufeff'):
             df_raw.rename(columns={df_raw.columns[0]: df_raw.columns[0].replace('\ufeff', '')}, inplace=True)
        
        df_raw['date'] = pd.to_datetime(df_raw['date'], format='%Y/%m')
        df_raw['year'] = df_raw['date'].dt.year
        df_raw['month'] = df_raw['date'].dt.month
        df_raw['focuses'] = pd.to_numeric(df_raw['focuses'], errors='coerce').fillna(0).astype(int)
        for col in ['uf', 'class', 'year', 'month', 'focuses']:
            if col not in df_raw.columns:
                df_raw[col] = pd.Series(dtype='object' if col in ['uf', 'class'] else 'int')
        return df_raw
    except FileNotFoundError:
        return pd.DataFrame(columns=['date', 'class', 'focuses', 'uf', 'year', 'month'])


def query_hf_api(payload_dict, api_token_val):
    if not api_token_val:
        st.error("Token da API Hugging Face não configurado no arquivo .env (HF_API_TOKEN).")
        return None
    headers = {"Authorization": f"Bearer {api_token_val}", "Content-Type": "application/json"}
    json_payload = json.dumps(payload_dict)

    try:
        response = requests.post(HF_API_URL, headers=headers, data=json_payload.encode('utf-8'), timeout=45)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("Timeout ao tentar conectar com a API Hugging Face.")
        return None
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 503 and "model" in response.text.lower() and "is currently loading" in response.text.lower():
            return "LOADING"
        st.error(f"Erro HTTP na API Hugging Face ({response.status_code}) para o modelo {HF_MODEL_ID}: {response.text[:500]}")
        return None
    except Exception as e:
        st.error(f"Erro ao conectar com a API Hugging Face: {e}")
        return None

# --- Load Initial Data and Models ---
sklearn_model_pipeline, unique_classes_loaded, unique_ufs_loaded, model_feature_columns = load_sklearn_model_assets(MODEL_ARTIFACTS_PATHS)
df_display = load_fire_data(DATA_FILE_PATH)

assets_loaded_flags = {
    "sklearn_model": sklearn_model_pipeline is not None,
    "unique_classes": unique_classes_loaded is not None,
    "unique_ufs": unique_ufs_loaded is not None,
    "model_features": model_feature_columns is not None,
    "dataset": not df_display.empty or ('focuses' in df_display.columns)
}
all_sklearn_assets_loaded = all(assets_loaded_flags[k] for k in ["sklearn_model", "unique_classes", "unique_ufs", "model_features"])

# --- Sidebar ---
st.sidebar.title("🇧🇷🔥 Navegação Global Solution")
app_mode = st.sidebar.selectbox(
    "Escolha a seção:",
    ["Introdução", "Explicação do Projeto", "1. Análise Descritiva", "2. Análise Preditiva (Focos)", "Bônus: Consulta Inteligente com IA"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Turma:** 2TIAPY")
st.sidebar.markdown("**Disciplina:** Front End & Mobile Development")
st.sidebar.markdown("---")
st.sidebar.markdown("**Integrantes:**")
st.sidebar.markdown("Alan Maximiano RM557088")
st.sidebar.markdown("André Rovai RM555848")
st.sidebar.markdown("Leonardo Zago RM558691")
st.sidebar.markdown("---")
st.sidebar.markdown("### Informações da IA (Bônus)")
st.sidebar.caption(f"Modelo de IA usado: `{HF_MODEL_ID}`")
if not HF_API_TOKEN:
    st.sidebar.warning("Token da API Hugging Face (HF_API_TOKEN) não encontrado no arquivo `.env`.", icon="⚠️")

# --- Plotting Functions for Descriptive Analysis ---
def plot_focuses_over_time(df_plot, title, x_col, y_col='focuses', hue_col=None, plot_type='line', is_horizontal=False):
    if df_plot is None or df_plot.empty or x_col not in df_plot.columns or y_col not in df_plot.columns:
        st.caption(f"Dados insuficientes ou colunas ausentes para gerar o gráfico: {title}")
        return

    fig, ax = plt.subplots(figsize=(12, 7 if is_horizontal else 6))
    legend_status = False if (hue_col == x_col or hue_col == y_col) else (True if hue_col else None)

    if plot_type == 'line_bar':
        sns.lineplot(x=x_col, y=y_col, data=df_plot, ax=ax, marker='o', color='dodgerblue', hue=hue_col, legend=legend_status)
        sns.barplot(x=x_col, y=y_col, data=df_plot, ax=ax, color='skyblue', alpha=0.6, hue=hue_col, legend=False)
    elif plot_type == 'bar':
        sns.barplot(x=x_col, y=y_col, data=df_plot, ax=ax, palette="magma", hue=hue_col, legend=legend_status)
    elif plot_type == 'barh':
        sns.barplot(x=y_col, y=x_col, data=df_plot, ax=ax, palette="crest", orient='h', hue=hue_col, legend=legend_status)
    
    ax.set_title(title)
    ax.set_xlabel(x_col.capitalize())
    ax.set_ylabel('Total de Focos' if y_col == 'focuses' else y_col.capitalize())
    
    formatter = mticker.StrMethodFormatter('{x:,.0f}')
    if is_horizontal:
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    st.pyplot(fig)

# --- Main Application Logic ---
def display_introduction():
    st.title("🇧🇷🔥 Global Solution: Análise e Previsão de Focos de Incêndio no Brasil")
    st.markdown("""
    Bem-vindo à aplicação interativa para análise e previsão de focos de incêndio no Brasil,
    utilizando dados do [TerraBrasilis (INPE)](https://terrabrasilis.dpi.inpe.br/ಫ್ホーム/).
    Este projeto emprega um modelo de Machine Learning (Random Forest Regressor) para prever
    a quantidade de focos de incêndio com base em informações temporais, tipo de área e localização (UF).
    A seção bônus utiliza um modelo de Linguagem Grande (LLM) open-source acessado via API para responder perguntas sobre os dados.

    **Funcionalidades:**
    - **Análise Descritiva:** Explore visualizações e estatísticas do dataset de focos de incêndio.
    - **Análise Preditiva:** Insira dados como ano, mês, tipo de área e UF para obter uma previsão do número de focos.
    - **Consulta Inteligente com IA:** Faça perguntas em linguagem natural sobre os dados e obtenha respostas geradas por uma IA.

    Utilize o menu na barra lateral para navegar entre as seções.
    """)
    # Image was removed as per previous request.

def display_project_explanation():
    st.title("💡 Explicação do Projeto: Global Solution FIAP")
    st.markdown("---")
    st.subheader("O que é a Global Solution?")
    st.markdown("""
    A **Global Solution** é um componente crucial da avaliação semestral na FIAP, integrando os conhecimentos
    adquiridos em diversas disciplinas do semestre em um projeto prático e desafiador. O objetivo é
    incentivar os alunos a aplicarem teorias e ferramentas tecnológicas na resolução de problemas
    complexos, muitas vezes com relevância social, ambiental ou de mercado.

    Este formato de avaliação promove o desenvolvimento de habilidades como:
    - Pensamento crítico e resolução de problemas.
    - Trabalho em equipe (embora este projeto seja individual ou em pequenos grupos).
    - Gerenciamento de projetos e prazos.
    - Aplicação prática de tecnologias emergentes.
    - Comunicação e apresentação de resultados.
    """)

    st.subheader("Este Projeto: Análise e Previsão de Focos de Incêndio no Brasil")
    st.markdown("""
    No contexto da disciplina de **Front End & Mobile Development** (e potencialmente outras relevantes para o projeto),
    este projeto visa aplicar os conceitos de desenvolvimento de aplicações web interativas, análise de dados e
    machine learning para abordar a questão crítica dos focos de incêndio no Brasil.

    **Objetivos Específicos do Projeto:**
    1.  **Coleta e Preparação de Dados:** Utilizar dados públicos sobre focos de incêndio no Brasil, como os
        disponibilizados pelo INPE (Instituto Nacional de Pesquisas Espaciais) através da plataforma TerraBrasilis.
        Esta etapa envolve o tratamento, limpeza e transformação dos dados para que se tornem adequados para análise e modelagem.
    2.  **Análise Descritiva:** Desenvolver uma seção interativa na aplicação Streamlit que permita aos usuários
        explorar os dados históricos de focos de incêndio. Isso inclui a visualização de tendências temporais (anuais, mensais),
        distribuição geográfica (por UF), e a identificação das classes de área mais afetadas, através de gráficos e tabelas dinâmicas.
    3.  **Análise Preditiva:** Construir e treinar um modelo de Machine Learning (especificamente, um modelo de Regressão como o Random Forest Regressor)
        capaz de estimar o número de focos de incêndio. As previsões são baseadas em características como data (ano, mês),
        estado (UF) e o tipo de área onde o fogo ocorre (classe).
    4.  **Desenvolvimento da Aplicação Web com Streamlit:** Criar uma interface de usuário (UI) que seja amigável,
        intuitiva e informativa. A aplicação deve apresentar de forma clara os resultados das análises descritivas
        e permitir que os usuários insiram dados para obter previsões do modelo de Machine Learning.
    5.  **Bônus - Consulta Inteligente com IA:** Integrar um modelo de linguagem grande (LLM) open-source,
        acessado via API (Hugging Face Inference API), para permitir que os usuários façam perguntas em linguagem natural
        sobre os dados de incêndios, oferecendo uma forma mais flexível e interativa de extrair insights.

    **Relevância e Impacto Potencial:**
    Os incêndios florestais e em outras formas de vegetação são uma preocupação ambiental, social e econômica significativa
    no Brasil, com impactos na biodiversidade, saúde pública, agricultura e mudanças climáticas.
    Ferramentas que auxiliem na compreensão desses fenômenos, na identificação de padrões e na previsão de ocorrências
    podem ser extremamente valiosas para órgãos de fiscalização ambiental (como IBAMA, ICMBio), defesa civil,
    corpos de bombeiros, pesquisadores e para a conscientização da população em geral.
    Este projeto, ao mesmo tempo que cumpre os rigorosos requisitos acadêmicos da Global Solution da FIAP,
    busca demonstrar como a tecnologia pode ser aplicada para gerar insights e ferramentas úteis para
    enfrentar desafios do mundo real.
    """)
    st.markdown("---")

def display_descriptive_analysis():
    st.header("1. Análise Descritiva dos Dados de Focos de Incêndio")

    if not assets_loaded_flags["dataset"]:
        st.error(f"Erro: Arquivo de dados '{DATA_FILE_PATH}' não encontrado ou o arquivo está vazio. A análise descritiva não pode ser exibida.")
        return

    ufs_options = sorted(unique_ufs_loaded) if assets_loaded_flags["unique_ufs"] and unique_ufs_loaded else (sorted(list(df_display['uf'].unique())) if 'uf' in df_display.columns else [])
    classes_options = sorted(unique_classes_loaded) if assets_loaded_flags["unique_classes"] and unique_classes_loaded else (sorted(list(df_display['class'].unique())) if 'class' in df_display.columns else [])

    if not ufs_options: st.warning("Opções de UF não disponíveis (nem do joblib, nem do dataset). Filtro de UF desabilitado.", icon="⚠️")
    if not classes_options: st.warning("Opções de Classe não disponíveis (nem do joblib, nem do dataset). Filtro de Classe desabilitado.", icon="⚠️")
    
    if not assets_loaded_flags["unique_ufs"] or not assets_loaded_flags["unique_classes"]:
         if assets_loaded_flags["dataset"]: 
            st.sidebar.warning("Assets de filtro (.joblib) não encontrados. Usando opções do dataset, se disponíveis.", icon="⚠️")

    st.markdown("### Visão Geral do Dataset (Últimos 5 registros)")
    st.dataframe(df_display.tail(5), height=210, use_container_width=True)

    st.markdown("### Filtros para Análise")
    final_ufs_options = ['Todos'] + ufs_options
    final_classes_options = ['Todos'] + classes_options
    default_ufs = ['Todos'] if 'Todos' in final_ufs_options else (final_ufs_options[:1] if final_ufs_options else [])
    default_classes = ['Todos'] if 'Todos' in final_classes_options else (final_classes_options[:1] if final_classes_options else [])
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1: selected_uf = st.multiselect("UF:", options=final_ufs_options, default=default_ufs)
    with col_filter2: selected_class = st.multiselect("Classe:", options=final_classes_options, default=default_classes)

    filtered_df = df_display.copy() 
    if ufs_options and 'Todos' not in selected_uf : filtered_df = filtered_df[filtered_df['uf'].isin(selected_uf)]
    if classes_options and 'Todos' not in selected_class: filtered_df = filtered_df[filtered_df['class'].isin(selected_class)]

    if filtered_df is None or filtered_df.empty: 
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
        return
    
    st.markdown(f"### Análise para UF(s): `{', '.join(selected_uf)}` e Classe(s): `{', '.join(selected_class)}`")

    required_plot_cols = ['year', 'month', 'focuses', 'uf', 'class']
    if not all(col in filtered_df.columns for col in required_plot_cols):
        st.warning("Dados filtrados não contêm todas as colunas necessárias para os gráficos ('year', 'month', 'focuses', 'uf', 'class').")
    else:
        plot_focuses_over_time(filtered_df.groupby('year')['focuses'].sum().reset_index(), 'Total de Focos por Ano', 'year', plot_type='line_bar')
        plot_focuses_over_time(filtered_df.groupby('month')['focuses'].sum().reset_index(), 'Sazonalidade: Total de Focos por Mês', 'month', plot_type='bar', hue_col='month')

        if ('Todos' in selected_uf or len(selected_uf) > 1) and ufs_options: # Check ufs_options has content
            if not filtered_df.empty and 'uf' in filtered_df.columns and 'focuses' in filtered_df.columns:
                top_states_filt = filtered_df.groupby('uf')['focuses'].sum().nlargest(10).reset_index()
                if not top_states_filt.empty: plot_focuses_over_time(top_states_filt, 'Top Estados por Total de Focos', 'uf', plot_type='barh', hue_col='uf', is_horizontal=True)
        
        if ('Todos' in selected_class or len(selected_class) > 1) and classes_options: # Check classes_options has content
            if not filtered_df.empty and 'class' in filtered_df.columns and 'focuses' in filtered_df.columns:
                focuses_by_class_filt = filtered_df.groupby('class')['focuses'].sum().sort_values(ascending=False).reset_index()
                if not focuses_by_class_filt.empty: plot_focuses_over_time(focuses_by_class_filt, 'Focos por Tipo de Área', 'class', plot_type='barh', hue_col='class', is_horizontal=True)

    st.markdown("---"); st.markdown("### Mais Análises Tabulares (Dados Filtrados)")
    if not all(col in filtered_df.columns for col in required_plot_cols):
        st.warning("Dados filtrados não contêm todas as colunas necessárias para as tabelas adicionais.")
    else:
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.markdown("##### Top 3 UFs (Média Focos/Registro)")
            avg_focus_uf = filtered_df.groupby('uf')['focuses'].mean().nlargest(3).reset_index()
            st.dataframe(avg_focus_uf.style.format({'focuses': "{:.1f}"}), height=150, use_container_width=True)
            st.markdown("##### Ano com Mais Focos")
            year_summary = filtered_df.groupby('year')['focuses'].sum()
            if not year_summary.empty:
                year_most_focuses = year_summary.idxmax(); val_most_focuses = year_summary.max()
                st.metric(label=f"Ano: {year_most_focuses}", value=f"{val_most_focuses:,.0f} focos")
            else: st.caption("N/A")
        with col_t2:
            st.markdown("##### Top 3 Classes (Média Focos/Registro)")
            avg_focus_class = filtered_df.groupby('class')['focuses'].mean().nlargest(3).reset_index()
            st.dataframe(avg_focus_class.style.format({'focuses': "{:.1f}"}), height=150, use_container_width=True)
            st.markdown("##### Ano com Menos Focos")
            year_summary = filtered_df.groupby('year')['focuses'].sum()
            if not year_summary.empty:
                year_least_focuses = year_summary.idxmin(); val_least_focuses = year_summary.min()
                st.metric(label=f"Ano: {year_least_focuses}", value=f"{val_least_focuses:,.0f} focos")
            else: st.caption("N/A")
        with col_t3:
            st.markdown("##### Focos nos Últimos 6 Meses (Total)")
            if 'date' in filtered_df.columns and not filtered_df.empty:
                last_date = filtered_df['date'].max()
                six_months_ago = last_date - pd.DateOffset(months=5) 
                recent_focuses = filtered_df[filtered_df['date'] >= six_months_ago]
                if not recent_focuses.empty:
                    recent_summary = recent_focuses.groupby(recent_focuses['date'].dt.to_period('M'))['focuses'].sum().reset_index()
                    recent_summary['date'] = recent_summary['date'].astype(str)
                    st.dataframe(recent_summary.tail(6).sort_values(by='date', ascending=False), height=220, use_container_width=True)
                else: st.caption("N/A para últimos 6 meses.")
            else: st.caption("N/A")
            st.markdown("##### Média Geral de Focos/Registro")
            avg_geral = filtered_df['focuses'].mean()
            st.metric(label="Média Geral (filt.)", value=f"{avg_geral:,.1f} focos")

        st.markdown("##### Focos por Ano e Mês (Pivot)")
        pivot_ym = filtered_df.groupby(['year', 'month'])['focuses'].sum().unstack(fill_value=0)
        st.dataframe(pivot_ym.style.format("{:,.0f}"), height=300, use_container_width=True)
        st.markdown("##### Top 5 Combinações UF/Classe (Total Focos)")
        top_uf_class = filtered_df.groupby(['uf', 'class'])['focuses'].sum().nlargest(5).reset_index()
        st.dataframe(top_uf_class.style.format({'focuses': "{:,.0f}"}), height=220, use_container_width=True)
        
        st.markdown("##### Variação Percentual Mensal (Total Geral de Focos - Dataset Completo)")
        if 'date' in df_display.columns and 'focuses' in df_display.columns:
            monthly_total_full_ds = df_display.groupby(df_display['date'].dt.to_period('M'))['focuses'].sum()
            if len(monthly_total_full_ds) > 1:
                monthly_pct_change = monthly_total_full_ds.pct_change().fillna(0) * 100
                monthly_pct_change_df = monthly_pct_change.reset_index()
                monthly_pct_change_df['date'] = monthly_pct_change_df['date'].astype(str)
                st.line_chart(monthly_pct_change_df.set_index('date')['focuses'].rename("Variação % Mensal"))
            else: st.caption("Dados insuficientes para calcular variação mensal.")

        st.markdown("##### Estado com Mais Focos por Tipo de Classe (Top 5 Classes)")
        if 'class' in filtered_df.columns and 'uf' in filtered_df.columns and 'focuses' in filtered_df.columns:
            if not filtered_df.empty:
                idx = filtered_df.groupby(['class'])['focuses'].transform('max') == filtered_df['focuses']
                top_class_per_uf = filtered_df[idx].drop_duplicates(subset=['class'], keep='first')[['class', 'uf', 'focuses']]
                top_classes_overall = filtered_df.groupby('class')['focuses'].sum().nlargest(5).index
                st.dataframe(top_class_per_uf[top_class_per_uf['class'].isin(top_classes_overall)].sort_values(by='focuses', ascending=False), height=220, use_container_width=True)
            else: st.caption("N/A")

        st.markdown("##### Classe com Mais Focos por Estado (Top 5 Estados)")
        if 'uf' in filtered_df.columns and 'class' in filtered_df.columns and 'focuses' in filtered_df.columns:
            if not filtered_df.empty:
                idx_cl = filtered_df.groupby(['uf'])['focuses'].transform('max') == filtered_df['focuses']
                top_uf_per_class = filtered_df[idx_cl].drop_duplicates(subset=['uf'], keep='first')[['uf', 'class', 'focuses']]
                top_ufs_overall = filtered_df.groupby('uf')['focuses'].sum().nlargest(5).index
                st.dataframe(top_uf_per_class[top_uf_per_class['uf'].isin(top_ufs_overall)].sort_values(by='focuses', ascending=False), height=220, use_container_width=True)
            else: st.caption("N/A")


def display_predictive_analysis():
    st.header("2. Previsão do Número de Focos de Incêndio")
    if not all_sklearn_assets_loaded:
        st.error("Erro: Arquivos do modelo scikit-learn (.joblib) não encontrados ou incompletos. Execute o notebook de treinamento primeiro. A análise preditiva não pode ser realizada.")
        return
    if not assets_loaded_flags["dataset"]:
        st.error("Erro: Arquivo de dados não carregado. A análise preditiva não pode ser realizada.")
        return

    st.markdown("Insira os dados para prever o número de focos de incêndio:")
    current_year = pd.Timestamp.now().year
    
    year_min_val = df_display['year'].min() if 'year' in df_display.columns and not df_display['year'].empty else current_year - 5
    year_range = list(range(year_min_val, current_year + 6))
    default_year_idx = year_range.index(current_year) if current_year in year_range else len(year_range) - 6 
    
    classes_for_select = sorted(unique_classes_loaded) if unique_classes_loaded else []
    ufs_for_select = sorted(unique_ufs_loaded) if unique_ufs_loaded else []

    if not classes_for_select or not ufs_for_select:
        st.error("Opções de Classe ou UF não carregadas dos arquivos .joblib. Não é possível fazer a previsão.")
        return

    col1, col2 = st.columns(2)
    with col1:
        input_year = st.selectbox("Ano:", year_range, index=default_year_idx)
        input_class = st.selectbox("Classe da Área:", classes_for_select)
    with col2:
        input_month = st.selectbox("Mês:", list(range(1, 13)), index=pd.Timestamp.now().month -1 )
        input_uf = st.selectbox("Estado (UF):", ufs_for_select)
    
    if st.button("🔥 Prever Número de Focos"):
        input_data_dict = {'year': [input_year], 'month': [input_month], 'class': [input_class], 'uf': [input_uf]}
        input_df = pd.DataFrame(input_data_dict)[model_feature_columns] 
        predicted_focuses = sklearn_model_pipeline.predict(input_df)
        
        st.subheader("Resultado da Predição:")
        st.info(f"Número estimado de focos de incêndio: **{max(0, round(predicted_focuses[0]))}**")
        
        if 'month' in df_display.columns and 'uf' in df_display.columns and 'class' in df_display.columns:
            hist_data = df_display[(df_display['month'] == input_month) & (df_display['uf'] == input_uf) & (df_display['class'] == input_class)]
            if not hist_data.empty:
                avg_hist = hist_data['focuses'].mean()
                max_hist = hist_data['focuses'].max()
                st.markdown(f"**Contexto Histórico para {input_uf} - {input_class} no mês {input_month}:**\n"
                            f"- Média de focos em anos anteriores: {avg_hist:.0f}\n"
                            f"- Máximo de focos em anos anteriores: {max_hist:.0f}")
            else:
                st.markdown("Nenhum dado histórico exato encontrado para esta combinação.")
        else:
            st.caption("Colunas necessárias para contexto histórico não encontradas.")

def display_ai_query():
    st.header("💡 Bônus: Consulta Inteligente com IA Open-Source (via API)")
    
    if not assets_loaded_flags["dataset"]:
        st.warning("Dataset não carregado. A consulta inteligente não pode funcionar.")
        return
    if not HF_API_TOKEN:
        st.warning("Token da API Hugging Face (HF_API_TOKEN) não encontrado/configurado no arquivo `.env`. Esta funcionalidade está desabilitada.")
        return

    st.markdown("""
    Faça uma pergunta em linguagem natural sobre os focos de incêndio no Brasil.
    A IA tentará responder com base nos dados disponíveis. <br>
    *Nota: A IA pode ocasionalmente cometer erros ou "alucinar" informações se a pergunta for ambígua ou os dados limitados. Verifique informações críticas.*
    Exemplos:
    - "Qual o total de focos de incêndio no estado do Amazonas em 2023?"
    - "Quais os tipos de área (classes) com mais focos historicamente?"
    - "Existe alguma tendência de aumento de focos em Roraima nos últimos anos?"
    """, unsafe_allow_html=True)

    user_query = st.text_input("Sua pergunta:", key="genai_query_api")

    if user_query:
        with st.spinner("Consultando a IA... Isso pode levar alguns segundos."):
            df_sample_for_llm = df_display[['year', 'month', 'uf', 'class', 'focuses']].sample(min(3, len(df_display)), random_state=42) if not df_display.empty else pd.DataFrame()
            sample_str = df_sample_for_llm.to_string(index=False, max_rows=3) if not df_sample_for_llm.empty else "Nenhum dado de exemplo disponível."

            context_prompt = (
                "Você é um assistente de IA para análise de dados de focos de incêndio no Brasil.\n"
                f"Colunas do DataFrame: {', '.join(df_display.columns)}.\n"
                "'focuses' é o número de focos. 'uf' é o estado. 'year' e 'month' são ano e mês.\n"
                f"Exemplo de dados:\n{sample_str}\n\n"
                "INSTRUÇÃO IMPORTANTE: Responda SOMENTE à pergunta do usuário. Não repita a pergunta. Não use tags como '[USUÁRIO]' ou '[ASSISTENTE]' na sua resposta final. "
                "Seja direto e informativo, baseando-se nos dados. Se a informação exata não estiver nos exemplos ou for muito específica para deduzir, "
                "indique que a análise detalhada dos dados completos seria necessária ou que a informação não está presente nos exemplos."
            )
            full_prompt = f"<s>[INST] {context_prompt} \n\nPergunta do Usuário: {user_query} \n\nSua Resposta Direta: [/INST]"
            
            payload_dict = {
                "inputs": full_prompt,
                "parameters": {"max_new_tokens": 200, "temperature": 0.4, "top_p": 0.9, "do_sample": True, "repetition_penalty": 1.15},
                "options": {"wait_for_model": True, "use_cache": False}
            }
            api_response = query_hf_api(payload_dict, HF_API_TOKEN)
            
            st.markdown("---"); st.subheader("Resposta da IA:")
            if api_response == "LOADING":
                st.warning("O modelo de IA está carregando. Tente novamente em instantes.")
            elif isinstance(api_response, list) and api_response and "generated_text" in api_response[0]:
                full_gen_text = api_response[0]["generated_text"]
                answer_part = full_gen_text
                if "[/INST]" in full_gen_text:
                    answer_part = full_gen_text.split("[/INST]", 1)[-1].strip()
                answer_part = answer_part.replace("[USUÁRIO]", "").replace("[ASSISTENTE]", "").strip()
                st.info(answer_part if answer_part else "A IA não forneceu uma resposta utilizável ou a resposta estava vazia após a limpeza.")
            elif isinstance(api_response, dict) and "error" in api_response:
                st.error(f"Erro da API: {api_response['error']}")
                if "estimated_time" in api_response: st.info(f"Tempo estimado para o modelo carregar: {api_response['estimated_time']:.0f}s.")
            else:
                st.error("Não foi possível obter resposta da IA ou resposta em formato inesperado.")
                if api_response: st.json(api_response)
    st.markdown("---")
    st.caption(f"Utiliza a API de Inferência da Hugging Face com o modelo `{HF_MODEL_ID}`. Requer token em `.env`.")

# --- Page Routing ---
if not assets_loaded_flags["dataset"] and app_mode not in ["Introdução", "Explicação do Projeto"]:
    st.error(f"Erro fatal: Arquivo de dados principal '{DATA_FILE_PATH}' não encontrado ou vazio. A aplicação não pode continuar com esta seção.")
else:
    if not all_sklearn_assets_loaded and app_mode in ["1. Análise Descritiva", "2. Análise Preditiva (Focos)"]:
        st.sidebar.error("Arquivos de modelo (.joblib) podem estar faltando. Execute o notebook de treinamento.", icon="🚨")

    if app_mode == "Introdução":
        display_introduction()
    elif app_mode == "Explicação do Projeto":
        display_project_explanation()
    elif app_mode == "1. Análise Descritiva":
        display_descriptive_analysis()
    elif app_mode == "2. Análise Preditiva (Focos)":
        display_predictive_analysis() 
    elif app_mode == "Bônus: Consulta Inteligente com IA":
        display_ai_query()

# --- Footer ---
st.markdown("---")
st.markdown("Desenvolvido para a GS - FIAP")
st.markdown("Integrantes:")
st.markdown("Alan Maximiano RM557088")
st.markdown("André Rovai RM555848")
st.markdown("Leonardo Zago RM558691")