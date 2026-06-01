# Tennis Predicter: ATP Tennis Favorite-Wins Prediction

## Project Overview

This project is an Introduction to Artificial Intelligence
application that uses historical ATP tennis match data to predict whether the
favorite player wins a match.

The model does not try to predict a player's career, tournament result, or exact
score. It answers one focused binary classification question:

```text
Will the favorite player win this ATP match?
```

In this project, the favorite is defined as the player with the better ATP
ranking. A lower ranking number means a stronger ranking position, so rank 2 is
considered favored over rank 20.

The output of the model is a probability between 0 and 1:

- A value close to `1` means the model believes the favorite is likely to win.
- A value close to `0` means the model believes an upset is more likely.

This makes the project useful as a demonstration of supervised learning,
feature engineering, neural networks, and model evaluation with a confusion
matrix.

## Problem Definition

The task is binary classification.

The target variable is:

- `1`: the favorite player wins.
- `0`: the underdog player wins.

For every match, the project compares the two players, identifies which one is
the favorite, builds numerical features from the match information, and trains a
neural network to learn patterns that are associated with favorite wins and
upsets.

This is a predictive model, but it should not be understood as "seeing the
future." It estimates probabilities from historical data. Its prediction depends
on the information given to it, such as rankings, ATP points, playing surface,
and recent form.

## Dataset

The dataset used by this project is an ATP tennis CSV file named:

```text
atp_tennis.csv
```

The expected dataset source is Kaggle:

```text
ATP Tennis 2000-2026 Daily Update
```

The project expects columns such as:

- `Date`
- `Rank_1`
- `Rank_2`
- `Player_1`
- `Player_2`
- `Winner`
- `Surface`
- `Pts_1`
- `Pts_2`
- optional: `Surface_Encoded`

Before training, the project removes duplicate rows and rows that are missing
required values. It also sorts matches by date so that recent-form features are
computed in chronological order.

## Feature Engineering

The model receives five numeric input features.

### 1. Rank Difference

```text
Rank_Diff = favorite rank - underdog rank
```

Because the favorite has the lower rank number, this value is usually negative.
For example, if the favorite is rank 5 and the underdog is rank 30:

```text
Rank_Diff = 5 - 30 = -25
```

This feature tells the model how large the ranking gap is between the two
players.

### 2. Surface Encoding

```text
Surface_Encoded
```

The court surface is converted into a number so the neural network can use it.
Typical surfaces include hard, clay, and grass. This matters because some
players perform better on certain surfaces.

### 3. Favorite Recent Form

```text
Favorite_Form
```

This is the favorite player's recent win rate before the current match. The
default window is the previous 30 matches. Only matches that happened before the
current row are used, which helps avoid data leakage.

### 4. Underdog Recent Form

```text
Underdog_Form
```

This is the underdog player's recent win rate before the current match, using
the same chronological method as the favorite's form.

### 5. Points Difference

```text
Pts_Diff = favorite ATP points - underdog ATP points
```

ATP points provide another measure of player strength. This feature helps the
model compare the players beyond only their ranking numbers.

## Model Design

The project uses a Keras neural network, specifically a multi-layer perceptron
(MLP). An MLP is a feed-forward artificial neural network made from dense layers.
It is appropriate here because the input is a small table of numeric features.

The architecture is:

- Input layer: 5 features.
- Hidden layer 1: 12 neurons with ReLU activation and He normal initialization.
- Hidden layer 2: 8 neurons with ReLU activation and He normal initialization.
- Output layer: 1 neuron with sigmoid activation.

The model is compiled with:

- Loss function: `binary_crossentropy`
- Optimizer: Adam with learning rate `0.001`
- Metric: accuracy

Adam is used because it is an adaptive optimizer that changes the model weights
efficiently during training. It helps the neural network reduce the
`binary_crossentropy` loss by adjusting each weight step-by-step based on the
training errors. Adam is a common choice for MLP neural networks because it
usually trains smoothly without requiring much manual tuning.

The sigmoid output is important because it converts the final model score into a
probability-like value between 0 and 1. The project then uses a decision
threshold, usually `0.60`, to convert that probability into a class prediction:

- Probability >= 0.60: predict favorite wins.
- Probability < 0.60: predict upset.

## Training Workflow

The training script is:

```text
atp_mlp_keras.py
```

The workflow is:

1. Load the ATP CSV file.
2. Validate required columns.
3. Remove duplicate and incomplete rows.
4. Sort matches by date.
5. Build the five model features.
6. Create the binary target variable.
7. Standardize the feature values with `StandardScaler`.
8. Train the Keras MLP.
9. Evaluate the model with stratified K-fold cross-validation.
10. Save the final model, scaler, and metadata for the Streamlit app.

The scaler is saved because the model was trained on standardized values. During
prediction, new inputs must be transformed with the same scaler. If the scaler
is skipped, the model receives values in a different format from the one it
learned during training.

## Evaluation

The project evaluates the model with a confusion matrix and common
classification metrics.

The confusion matrix uses these labels:

- True Negative (TN): the model predicted an upset, and the match was an upset.
- False Positive (FP): the model predicted favorite wins, but the underdog won.
- False Negative (FN): the model predicted an upset, but the favorite won.
- True Positive (TP): the model predicted favorite wins, and the favorite won.

The project reports:

- Accuracy: the percentage of total correct predictions.
- Precision: when the model predicts favorite wins, how often it is correct.
- Recall: out of all real favorite wins, how many the model catches.
- F1 score: the balance between precision and recall.

The F1 score is calculated as:

```text
F1 = 2 * (precision * recall) / (precision + recall)
```

Accuracy alone can be misleading in this project because favorites usually win
more often than underdogs. A model could predict "favorite wins" almost every
time and still look reasonably accurate. The confusion matrix shows whether the
model is actually identifying both favorite wins and upsets.

## Baseline Comparison

The project compares the neural network against a simple baseline:

```text
Always predict that the favorite wins.
```

This baseline is important because it represents the obvious strategy. Since the
favorite already has the better ranking, a useful model should do more than
repeat this assumption. The MLP is more convincing if it improves over the
baseline in accuracy, F1 score, or its ability to detect upsets.

## Streamlit Application

The Streamlit app is located at:

```text
apps/confusion_matrix_app.py
```

The app demonstrates the trained model interactively. It can:

- Load the ATP dataset.
- Load the saved Keras model, scaler, and metadata.
- Show the confusion matrix.
- Display accuracy, precision, recall, and F1 score.
- Compare the MLP against the favorite-always-wins baseline.
- Show a simple feature-importance proxy from the neural network weights.
- Export the confusion matrix as a PDF.
- Predict win probabilities for a selected player-vs-player matchup.

For manual player predictions, the app identifies which selected player is the
favorite, builds the same five features used during training, scales them, and
then sends them to the saved model.

## Saved Artifacts

Training creates three important files:

```text
outputs/atp_model.keras
outputs/atp_scaler.joblib
outputs/atp_model_metadata.json
```

Their roles are:

- `atp_model.keras`: the trained Keras neural network.
- `atp_scaler.joblib`: the fitted `StandardScaler`.
- `atp_model_metadata.json`: information about feature order and training
  settings.

The model should be used together with the scaler and metadata. The metadata is
especially useful because the feature order must stay consistent between
training and prediction.

## Limitations

This project is a learning-focused AI model, not a professional betting system.
Several limitations should be considered:

- The model only uses five engineered features.
- It does not include injuries, fatigue, weather, travel, tournament pressure,
  head-to-head history, or betting odds.
- Player form is estimated from previous matches in the dataset.
- The model can be biased toward predicting favorite wins because favorites are
  often the majority class.
- The saved final model is trained on the full prepared dataset for use in the
  Streamlit demo, so cross-validation results are the better source for judging
  general performance.

These limitations do not make the project invalid. They show that the model is a
controlled classroom example of machine learning rather than a complete sports
forecasting system.

## Technologies Used

- Python: main programming language.
- TensorFlow / Keras: neural network training and saving.
- Pandas: CSV loading, cleaning, and feature preparation.
- NumPy: numerical arrays and feature calculations.
- scikit-learn: scaling, label encoding, cross-validation, and metrics.
- Matplotlib: confusion matrix plotting and PDF export.
- Streamlit: interactive app for evaluation and prediction.
- joblib: saving and loading the fitted scaler.
- Jupyter Notebook: optional interactive exploration workflow.

## Project Structure

```text
AIProject/
  atp_mlp_keras.py                  # CLI training and evaluation script
  atp_run.ipynb                     # Optional notebook workflow
  atp_tennis.csv                    # Local dataset file
  requirements.txt                  # Python dependencies

  apps/
    confusion_matrix_app.py         # Streamlit app

  src/ai_project/
    atp_features.py                 # Preprocessing and feature engineering
    model_artifacts.py              # Save/load model artifacts
    metrics/
      baselines.py                  # Favorite-wins baseline
      classification_metrics.py     # Accuracy, precision, recall, F1
      confusion_matrix.py           # Confusion matrix helpers
      plots.py                      # Confusion matrix plotting

  docs/
    PROJECT_GUIDE.md                # Step-by-step project guide

  tests/
    test_atp_features.py
    test_model_artifacts.py
    test_streamlit_app_structure.py
    test_training_dataset_loading.py
```

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

On Windows PowerShell, a virtual environment can be created and activated with:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train And Save The Model

Run:

```bash
python atp_mlp_keras.py atp_tennis.csv
```

This trains and evaluates the model, then saves the artifacts used by the
Streamlit app.

Optional PDF export:

```bash
python atp_mlp_keras.py atp_tennis.csv --export-pdf outputs/confusion_matrix.pdf
```

Useful training options:

```bash
python atp_mlp_keras.py atp_tennis.csv --epochs 100 --batch-size 64 --form-window 30
```

## Run The Streamlit App

On Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
streamlit run apps/confusion_matrix_app.py
```

On macOS/Linux:

```bash
PYTHONPATH=src streamlit run apps/confusion_matrix_app.py
```

In the app:

1. Confirm the CSV path.
2. Confirm the saved model, scaler, and metadata paths.
3. Click `Load saved model`.
4. Review the confusion matrix and metrics.
5. Use the player-vs-player prediction section.

## Run Tests

On Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -q
```

On macOS/Linux:

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
```

## Conclusion

This project shows how an AI model can be trained to make a focused sports
prediction from historical data. The model learns from rankings, ATP points,
surface, and recent form to estimate whether the favorite player is likely to
win. The Streamlit app then makes the results easier to inspect by showing the
confusion matrix, evaluation metrics, baseline comparison, feature importance,
and interactive matchup predictions.

---

# Tennis Predicter: Previsão de Vitória do Favorito no ATP Tennis

## Visão Geral do Projeto

Este projeto é uma aplicação de Introdução à Inteligência Artificial que usa
dados históricos de jogos de ténis ATP para prever se o jogador favorito ganha
uma partida.

O modelo não tenta prever a carreira de um jogador, o resultado de um torneio ou
o resultado exato do jogo. Responde a uma pergunta simples de classificação
binária:

```text
Will the favorite player win this ATP match?
```

Neste projeto, o favorito é definido como o jogador com melhor ranking ATP. Um
número de ranking mais baixo significa uma posição mais forte, por isso o jogador
com ranking 2 é considerado favorito contra o jogador com ranking 20.

A saída do modelo é uma probabilidade entre 0 e 1:

- Um valor próximo de `1` significa que o modelo acredita que o favorito tem mais
  probabilidade de ganhar.
- Um valor próximo de `0` significa que o modelo acredita que uma surpresa é mais
  provável.

Isto torna o projeto útil como demonstração de aprendizagem supervisionada,
engenharia de atributos, redes neuronais e avaliação de modelos com uma matriz
de confusão.

## Definição do Problema

A tarefa é uma classificação binária.

A variável alvo é:

- `1`: o jogador favorito ganha.
- `0`: o jogador não favorito ganha.

Para cada partida, o projeto compara os dois jogadores, identifica qual deles é
o favorito, constrói atributos numéricos a partir da informação do jogo e treina
uma rede neuronal para aprender padrões associados a vitórias do favorito e a
surpresas.

Este é um modelo preditivo, mas não deve ser entendido como algo que "vê o
futuro". O modelo estima probabilidades a partir de dados históricos. A previsão
depende da informação dada ao modelo, como rankings, pontos ATP, superfície de
jogo e forma recente.

## Conjunto de Dados

O conjunto de dados usado por este projeto é um ficheiro CSV de ténis ATP
chamado:

```text
atp_tennis.csv
```

A fonte esperada do conjunto de dados é o Kaggle:

```text
ATP Tennis 2000-2026 Daily Update
```

O projeto espera colunas como:

- `Date`
- `Rank_1`
- `Rank_2`
- `Player_1`
- `Player_2`
- `Winner`
- `Surface`
- `Pts_1`
- `Pts_2`
- opcional: `Surface_Encoded`

Antes do treino, o projeto remove linhas duplicadas e linhas com valores em
falta nas colunas necessárias. Também ordena os jogos por data para que os
atributos de forma recente sejam calculados em ordem cronológica.

## Engenharia de Atributos

O modelo recebe cinco atributos numéricos de entrada.

### 1. Diferença de Ranking

```text
Rank_Diff = ranking do favorito - ranking do não favorito
```

Como o favorito tem o número de ranking mais baixo, este valor é normalmente
negativo. Por exemplo, se o favorito for ranking 5 e o outro jogador for ranking
30:

```text
Rank_Diff = 5 - 30 = -25
```

Este atributo indica ao modelo quão grande é a diferença de ranking entre os
dois jogadores.

### 2. Codificação da Superfície

```text
Surface_Encoded
```

A superfície do campo é convertida num número para que a rede neuronal a possa
usar. Superfícies comuns incluem hard, clay e grass. Isto é importante porque
alguns jogadores têm melhor desempenho em certas superfícies.

### 3. Forma Recente do Favorito

```text
Favorite_Form
```

Esta é a taxa de vitórias recente do jogador favorito antes da partida atual. A
janela predefinida usa os 30 jogos anteriores. Apenas jogos que aconteceram
antes da linha atual são usados, o que ajuda a evitar fuga de dados.

### 4. Forma Recente do Não Favorito

```text
Underdog_Form
```

Esta é a taxa de vitórias recente do jogador não favorito antes da partida
atual, usando o mesmo método cronológico da forma do favorito.

### 5. Diferença de Pontos

```text
Pts_Diff = pontos ATP do favorito - pontos ATP do não favorito
```

Os pontos ATP dão outra medida da força dos jogadores. Este atributo ajuda o
modelo a comparar os jogadores para além do número do ranking.

## Desenho do Modelo

O projeto usa uma rede neuronal Keras, especificamente um perceptrão multicamada
(MLP). Um MLP é uma rede neuronal feed-forward composta por camadas densas. É
adequado aqui porque a entrada é uma pequena tabela de atributos numéricos.

A arquitetura é:

- Camada de entrada: 5 atributos.
- Camada escondida 1: 12 neurónios com ativação ReLU e inicialização He normal.
- Camada escondida 2: 8 neurónios com ativação ReLU e inicialização He normal.
- Camada de saída: 1 neurónio com ativação sigmoid.

O modelo é compilado com:

- Função de perda: `binary_crossentropy`
- Otimizador: Adam com taxa de aprendizagem `0.001`
- Métrica: accuracy

O Adam é usado porque é um otimizador adaptativo que altera os pesos do modelo de
forma eficiente durante o treino. Ajuda a rede neuronal a reduzir a perda
`binary_crossentropy`, ajustando cada peso passo a passo com base nos erros de
treino. O Adam é uma escolha comum para redes neuronais MLP porque costuma
treinar de forma estável sem exigir muita afinação manual.

A saída sigmoid é importante porque converte o resultado final do modelo num
valor semelhante a uma probabilidade entre 0 e 1. O projeto usa depois um limiar
de decisão, normalmente `0.60`, para converter essa probabilidade numa classe:

- Probabilidade >= 0.60: prever vitória do favorito.
- Probabilidade < 0.60: prever surpresa.

## Fluxo de Treino

O script de treino é:

```text
atp_mlp_keras.py
```

O fluxo é:

1. Carregar o ficheiro CSV ATP.
2. Validar as colunas necessárias.
3. Remover linhas duplicadas e incompletas.
4. Ordenar os jogos por data.
5. Construir os cinco atributos do modelo.
6. Criar a variável alvo binária.
7. Normalizar os valores dos atributos com `StandardScaler`.
8. Treinar o MLP em Keras.
9. Avaliar o modelo com validação cruzada estratificada K-fold.
10. Guardar o modelo final, o scaler e os metadados para a aplicação Streamlit.

O scaler é guardado porque o modelo foi treinado com valores normalizados. Durante
a previsão, novas entradas têm de ser transformadas com o mesmo scaler. Se o
scaler não for usado, o modelo recebe valores num formato diferente daquele que
aprendeu durante o treino.

## Avaliação

O projeto avalia o modelo com uma matriz de confusão e métricas comuns de
classificação.

A matriz de confusão usa estes rótulos:

- True Negative (TN): o modelo previu surpresa e o jogo foi uma surpresa.
- False Positive (FP): o modelo previu vitória do favorito, mas o não favorito
  ganhou.
- False Negative (FN): o modelo previu surpresa, mas o favorito ganhou.
- True Positive (TP): o modelo previu vitória do favorito e o favorito ganhou.

O projeto apresenta:

- Accuracy: a percentagem total de previsões corretas.
- Precision: quando o modelo prevê vitória do favorito, com que frequência está
  correto.
- Recall: de todas as vitórias reais do favorito, quantas o modelo identifica.
- F1 score: o equilíbrio entre precision e recall.

O F1 score é calculado assim:

```text
F1 = 2 * (precision * recall) / (precision + recall)
```

A accuracy isolada pode ser enganadora neste projeto porque os favoritos ganham
mais vezes do que os não favoritos. Um modelo poderia prever "vitória do
favorito" quase sempre e ainda assim parecer razoável. A matriz de confusão
mostra se o modelo está realmente a identificar tanto vitórias do favorito como
surpresas.

## Comparação com Baseline

O projeto compara a rede neuronal com um baseline simples:

```text
Always predict that the favorite wins.
```

Este baseline é importante porque representa a estratégia óbvia. Como o favorito
já tem melhor ranking, um modelo útil deve fazer mais do que repetir essa
suposição. O MLP é mais convincente se melhorar o baseline em accuracy, F1 score
ou capacidade de detetar surpresas.

## Aplicação Streamlit

A aplicação Streamlit está localizada em:

```text
apps/confusion_matrix_app.py
```

A aplicação demonstra o modelo treinado de forma interativa. Permite:

- Carregar o conjunto de dados ATP.
- Carregar o modelo Keras guardado, o scaler e os metadados.
- Mostrar a matriz de confusão.
- Mostrar accuracy, precision, recall e F1 score.
- Comparar o MLP com o baseline que prevê sempre vitória do favorito.
- Mostrar uma aproximação simples da importância dos atributos a partir dos
  pesos da rede neuronal.
- Exportar a matriz de confusão como PDF.
- Prever probabilidades de vitória para um confronto entre dois jogadores.

Para previsões manuais entre jogadores, a aplicação identifica qual dos
jogadores selecionados é o favorito, constrói os mesmos cinco atributos usados no
treino, normaliza-os e envia-os para o modelo guardado.

## Artefactos Guardados

O treino cria três ficheiros importantes:

```text
outputs/atp_model.keras
outputs/atp_scaler.joblib
outputs/atp_model_metadata.json
```

As suas funções são:

- `atp_model.keras`: a rede neuronal Keras treinada.
- `atp_scaler.joblib`: o `StandardScaler` ajustado.
- `atp_model_metadata.json`: informação sobre a ordem dos atributos e as
  configurações de treino.

O modelo deve ser usado juntamente com o scaler e os metadados. Os metadados são
especialmente úteis porque a ordem dos atributos tem de se manter consistente
entre treino e previsão.

## Limitações

Este projeto é um modelo de IA focado em aprendizagem, não um sistema
profissional de apostas. Devem ser consideradas várias limitações:

- O modelo usa apenas cinco atributos construídos.
- Não inclui lesões, fadiga, meteorologia, viagens, pressão do torneio,
  histórico direto entre jogadores ou odds.
- A forma dos jogadores é estimada a partir de jogos anteriores no conjunto de
  dados.
- O modelo pode ter tendência para prever vitórias do favorito porque os
  favoritos são frequentemente a classe maioritária.
- O modelo final guardado é treinado no conjunto de dados preparado completo
  para uso na demonstração Streamlit, por isso os resultados da validação
  cruzada são uma melhor fonte para avaliar a generalização.

Estas limitações não tornam o projeto inválido. Mostram que o modelo é um
exemplo controlado de machine learning para contexto académico, e não um sistema
completo de previsão desportiva.

## Tecnologias Usadas

- Python: linguagem de programação principal.
- TensorFlow / Keras: treino e armazenamento da rede neuronal.
- Pandas: carregamento, limpeza e preparação de atributos a partir do CSV.
- NumPy: arrays numéricos e cálculos de atributos.
- scikit-learn: normalização, codificação de rótulos, validação cruzada e
  métricas.
- Matplotlib: gráficos da matriz de confusão e exportação para PDF.
- Streamlit: aplicação interativa para avaliação e previsão.
- joblib: armazenamento e carregamento do scaler ajustado.
- Jupyter Notebook: fluxo opcional de exploração interativa.

## Estrutura do Projeto

```text
AIProject/
  atp_mlp_keras.py                  # Script CLI de treino e avaliação
  atp_run.ipynb                     # Workflow opcional em notebook
  atp_tennis.csv                    # Ficheiro local do conjunto de dados
  requirements.txt                  # Dependências Python

  apps/
    confusion_matrix_app.py         # Aplicação Streamlit

  src/ai_project/
    atp_features.py                 # Pré-processamento e engenharia de atributos
    model_artifacts.py              # Guardar/carregar artefactos do modelo
    metrics/
      baselines.py                  # Baseline de vitória do favorito
      classification_metrics.py     # Accuracy, precision, recall, F1
      confusion_matrix.py           # Funções da matriz de confusão
      plots.py                      # Gráficos da matriz de confusão

  docs/
    PROJECT_GUIDE.md                # Guia passo a passo do projeto

  tests/
    test_atp_features.py
    test_model_artifacts.py
    test_streamlit_app_structure.py
    test_training_dataset_loading.py
```

## Instalação

Instalar os pacotes Python necessários:

```bash
pip install -r requirements.txt
```

No Windows PowerShell, pode ser criado e ativado um ambiente virtual com:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Treinar e Guardar o Modelo

Executar:

```bash
python atp_mlp_keras.py atp_tennis.csv
```

Isto treina e avalia o modelo, e depois guarda os artefactos usados pela
aplicação Streamlit.

Exportação opcional para PDF:

```bash
python atp_mlp_keras.py atp_tennis.csv --export-pdf outputs/confusion_matrix.pdf
```

Opções úteis de treino:

```bash
python atp_mlp_keras.py atp_tennis.csv --epochs 100 --batch-size 64 --form-window 30
```

## Executar a Aplicação Streamlit

No Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
streamlit run apps/confusion_matrix_app.py
```

No macOS/Linux:

```bash
PYTHONPATH=src streamlit run apps/confusion_matrix_app.py
```

Na aplicação:

1. Confirmar o caminho do CSV.
2. Confirmar os caminhos do modelo guardado, do scaler e dos metadados.
3. Clicar em `Load saved model`.
4. Rever a matriz de confusão e as métricas.
5. Usar a secção de previsão jogador-vs-jogador.

## Executar Testes

No Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -q
```

No macOS/Linux:

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
```

## Conclusão

Este projeto mostra como um modelo de IA pode ser treinado para fazer uma
previsão desportiva focada a partir de dados históricos. O modelo aprende com
rankings, pontos ATP, superfície e forma recente para estimar se o jogador
favorito tem probabilidade de ganhar. A aplicação Streamlit facilita a análise
dos resultados ao mostrar a matriz de confusão, métricas de avaliação,
comparação com baseline, importância dos atributos e previsões interativas de
confrontos.
