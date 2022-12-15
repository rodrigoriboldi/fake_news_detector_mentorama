Este projeto corresponde ao projeto final do curso sobre ciência de dados na escola Mentorama.




# Classificador de Fake News

https://rodrigoriboldi-fake-news-detector-mentoram-streamlit-app-kir43f.streamlit.app/


Este programa corresponde ao projeto final do curso sobre ciência de dados na escola Mentorama.

Esse detector utiliza o resultado de quatro classificadores diferentes para dizer se uma notícia possui ou não a probabilidade de ser fake.

Os classificadores foram treinados com o conjunto de dados disponível neste link: https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view.

## Como utilizar 

Acesse o APP [neste link](https://rodrigoriboldi-fake-news-detector-mentoram-streamlit-app-kir43f.streamlit.app/) e cole o texto da notícia a ser classificada.


#### Modelos e Métricas
Os classificadores utilizados neste projeto foram treinados com um dataset de notícias em inglês e apresentaram bons resultados em um conjunto de dados de teste e um de validação, sendo então confiáveis para a utilização na classificação de notícias falsas.

Veja abaixo os resultados que cada classificador obteve com os dados de valiação.

Classificador 1: Random Forest
Accuracy: 0.91
Precision: 0.93
Recall: 0.89
AUC score: 0.91

Classificador 3: SVM
Accuracy: 0.92
Precision: 0.93
Recall: 0.92
AUC score: 0.92

Classificador 2: XGBost
Accuracy: 0.86
Precision: 0.88
Recall: 0.85
AUC score: 0.86

Classificador 4: Naive Bayes
Accuracy: 0.89
Precision: 0.95
Recall: 0.84
AUC score: 0.89


## Responsabilidade

Este é um projeto didático e seus resultados não devem ser utilizados como uma ferramenta de apoio para tomadas de decisões.

## Tecnologias Utilizadas

- Python
- Streamlit
