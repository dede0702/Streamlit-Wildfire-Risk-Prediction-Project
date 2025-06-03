Claro! Abaixo está uma versão aprimorada do README para o projeto [Streamlit-Wildfire-Risk-Prediction-Project](https://github.com/dede0702/Streamlit-Wildfire-Risk-Prediction-Project), incorporando melhores práticas de documentação e sugestões inspiradas em projetos semelhantes:

---

# 🔥 Previsão de Risco de Incêndios Florestais no Brasil com Streamlit

![Interface do Aplicativo](https://github.com/dede0702/Streamlit-Wildfire-Risk-Prediction-Project/blob/main/images/app_screenshot.png)

Este projeto utiliza técnicas de aprendizado de máquina para prever o risco de incêndios florestais no Brasil, oferecendo uma interface interativa desenvolvida com Streamlit. Ao combinar dados históricos de focos de incêndio com variáveis ambientais, a aplicação fornece previsões que auxiliam na prevenção e resposta a incêndios.

## 🌐 Acesse o Aplicativo

O aplicativo está disponível online:

👉 [gsfireanalysisbrazil.streamlit.app](https://gsfireanalysisbrazil.streamlit.app/)

## 🎯 Objetivo

Desenvolver uma ferramenta interativa que permita a visualização e previsão do risco de incêndios florestais no Brasil, auxiliando autoridades, pesquisadores e o público em geral na tomada de decisões e na conscientização sobre a prevenção de incêndios.

## 🧩 Funcionalidades

* **Interface Interativa**: Desenvolvida com Streamlit, permite fácil interação e visualização dos resultados.
* **Previsão de Risco**: Utiliza um modelo de aprendizado de máquina treinado para estimar o risco de incêndio com base em dados ambientais.
* **Visualização de Dados**: Gráficos e mapas que facilitam a compreensão dos padrões de incêndios.
* **Análise Exploratória**: Notebook com análises detalhadas dos dados utilizados.

## 🧠 Tecnologias Utilizadas

* **Python**: Linguagem principal para desenvolvimento.
* **Streamlit**: Framework para criação da interface web interativa.
* **scikit-learn**: Biblioteca de aprendizado de máquina para treinamento do modelo.
* **Pandas & NumPy**: Manipulação e análise de dados.
* **Joblib**: Serialização do modelo treinado.

## 📁 Estrutura do Projeto

```
Streamlit-Wildfire-Risk-Prediction-Project/
├── app_brazil.py
├── brazil_fire_focus_model.joblib
├── model_feature_columns.joblib
├── unique_classes.joblib
├── unique_ufs.joblib
├── all_dashboard-fires-month-02-06-2025-22_23_57.csv
├── gs_fire_analysis_brazil.ipynb
├── requirements.txt
└── images/
    └── app_screenshot.png
```

* `app_brazil.py`: Script principal da aplicação Streamlit.
* `brazil_fire_focus_model.joblib`: Modelo de aprendizado de máquina treinado.
* `model_feature_columns.joblib`: Colunas de características utilizadas pelo modelo.
* `unique_classes.joblib`: Classes únicas presentes nos dados.
* `unique_ufs.joblib`: Unidades federativas únicas presentes nos dados.
* `all_dashboard-fires-month-02-06-2025-22_23_57.csv`: Conjunto de dados utilizado para análise.
* `gs_fire_analysis_brazil.ipynb`: Notebook com análises exploratórias dos dados.
* `requirements.txt`: Lista de dependências do projeto.
* `images/`: Pasta contendo imagens utilizadas no README e na aplicação.

## 🚀 Como Executar Localmente

1. **Clone o repositório**:

   ```bash
   git clone https://github.com/dede0702/Streamlit-Wildfire-Risk-Prediction-Project.git
   cd Streamlit-Wildfire-Risk-Prediction-Project
   ```

2. **Crie um ambiente virtual e ative-o**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

3. **Instale as dependências**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação**:

   ```bash
   streamlit run app_brazil.py
   ```

5. **Acesse a aplicação** no navegador através do endereço exibido no terminal (geralmente `http://localhost:8501`).

## 📊 Dados e Modelo

* **Fonte dos Dados**: Dados históricos de focos de incêndio no Brasil.
* **Variáveis Consideradas**: Temperatura, umidade, velocidade do vento, entre outras.
* **Modelo Utilizado**: Classificador treinado com scikit-learn para prever o risco de incêndio.

## 👥 Equipe de Desenvolvimento

* **Alan Maximiano** - RM557088
* **André Rovai** - RM555848
* **Leonardo Zago** - RM558691

**Turma**: 2TIAPY
**Disciplina**: Front End & Mobile Development

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests com sugestões de melhorias, correções de bugs ou novas funcionalidades.

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

---

Se desejar, posso ajudá-lo a traduzir este README para o inglês ou adaptá-lo conforme suas necessidades específicas.
