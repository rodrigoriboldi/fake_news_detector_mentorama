"""
Projeto final do curso da Mentorama

Aluno: Rodrigo Martini Riboldi

Projeto: Classificador de notícias falsas

Aplicativo desenvolvido com o Streamlit
"""

# Importando bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import sklearn
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

# Configurando página
st.set_page_config(
    page_title="Detector de Fake News",
    layout="centered",
    initial_sidebar_state="collapsed")

# Importando modelo do Bag of Words
cv = pickle.load(open(dir_path + '/modelos/bag_of_words.sav', 'rb'))

# Importando modelos
rf_classifier = pickle.load(open(dir_path + '/modelos/random_forrest.sav', 'rb'))
xgb_classifier = pickle.load(open(dir_path + '/modelos/xgboost.sav', 'rb'))
svm_classifier = pickle.load(open(dir_path + '/modelos/svm.sav', 'rb'))
nb_classifier = pickle.load(open(dir_path + '/modelos/naive_bayes.sav', 'rb'))

# Importando exemplos
df_validation = pd.read_csv(dir_path + '/dados/df_validation.csv').drop(columns = ['Unnamed: 0', 'title'])

# Importando json com as métricas
with open(dir_path + '/metricas/model_metrics.json', 'r') as openfile:
    
    model_metrics_json = json.load(openfile)

# Titulos e textos
st.image("https://images.jota.info/wp-content/uploads/2021/12/fake-ga34dcd21f-1920-1-1536x1023.jpg")
st.title('Fake News detector')

st.caption('Projeto final do curso Cientista de Dados avançado da Mentorama \\\n Rodrigo Martini Riboldi')

with st.expander("Funcionamento:"):
    st.write('Este projeto corresponde ao projeto final do curso sobre ciência de dados na escola Mentorama. \\\n\\\n \
    Esse detector utiliza o resultado de quatro classificadores diferentes para dizer se uma notícia possui ou não a probabilidade de ser fake. \\\n\\\n \
    Os classificadores foram treinados com o conjunto de dados disponível neste link: https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view. \\\n\\\n \
    Você poderá consultar mais detalhes dos classificadores logo após os resultados. \\\n\\\n \
    A documentação do projeto pode ser conferida no seguinte link: https://github.com/rodrigoriboldi/fake_news_detector_mentorama')

# Área de interação com o usuário
st.header('Cole aqui o texto de uma notícia')

text = df_validation.text[1]

txt = st.text_area('O texto precisa ser em inglês! (Text must be in english)  ',
                   text,
                   height = 300,
                   key = 1)

# Tratamento do texto inserido
txt_vectorized = cv.transform(np.array([txt]))

# Classificação do texto inserido
rf_result = rf_classifier.predict(txt_vectorized)
xgb_result = xgb_classifier.predict(txt_vectorized)
svm_result = svm_classifier.predict(txt_vectorized)
nb_result = nb_classifier.predict(txt_vectorized)

rf_result_proba = rf_classifier.predict_proba(txt_vectorized)[0][1]*100
xgb_result_proba = xgb_classifier.predict_proba(txt_vectorized)[0][1]*100
svm_result_proba = svm_classifier.predict_proba(txt_vectorized)[0][1]*100
nb_result_proba = nb_classifier.predict_proba(txt_vectorized)[0][1]*100

result = rf_result+xgb_result+svm_result+nb_result

# Exibição do resultado
if result == 0:
    st.subheader('Essa notícia provavelmente não é fake!')
elif result == 1:
    st.subheader('Essa notícia possui poucas chances de ser fake.')
elif result == 2:
    st.subheader('Essa notícia pode ser fake.')
elif result == 3:
    st.subheader('Essa notícia possui muita chance de ser fake.')
else:
    st.subheader('Essa notícia provavelmente é fake!')

st.write('')
st.write('')
st.subheader('Resultados por classificador')
    
col1, col2, col3, col4 = st.columns(4)
with col1:
    if rf_result == 0:
        st.metric(label = 'Classificador 1', value='Real')
    else:
        st.metric(label = 'Classificador 1', value='Fake')

with col2:
    if xgb_result == 0:
        st.metric(label = 'Classificador 2', value='Real')
    else:
        st.metric(label = 'Classificador 2', value='Fake')

with col3:
    if svm_result == 0:
        st.metric(label = 'Classificador 3', value='Real')
    else:
        st.metric(label = 'Classificador 3', value='Fake')

with col4:
    if nb_result == 0:
        st.metric(label = 'Classificador 4', value='Real')
    else:
        st.metric(label = 'Classificador 4', value='Fake')

st.write('Probabilidade da notícia ser fake') 

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label = 'Classificador 1', value= str(round(rf_result_proba,2)) + '%')
with col2:
    st.metric(label = 'Classificador 2', value= str(round(xgb_result_proba,2)) + '%')
with col3:
    st.metric(label = 'Classificador 3', value= str(round(svm_result_proba,2)) + '%')
with col4:
    st.metric(label = 'Classificador 4', value= str(round(nb_result_proba,2)) + '%')   
    

    
with st.expander("Mais detalhes dos classificadores:"):
    st.write('Os classificadores utilizados neste projeto foram treinados com um dataset de notícias em inglês e apresentaram bons resultados em um conjunto de dados de teste e um de validação, sendo então confiáveis para a utilização na classificação de notícias falsas.')
    st.write('Veja abaixo os resultados que cada classificador obteve com os dados de valiação.')
             
    st.write('')
    
    col1, col2, = st.columns(2)
    with col1:
        st.write('Classificador 1: Random Forest \\\n \
                  Accuracy:  ' + str(round(model_metrics_json['RF Accuracy'],2)) + '  \\\n \
                  Precision: ' + str(round(model_metrics_json['RF Precision'],2)) + '  \\\n \
                  Recall:    ' + str(round(model_metrics_json['RF Recall'],2)) + '  \\\n \
                  AUC score: ' + str(round(model_metrics_json['RF AUC score'],2)) + '')

        st.write('')
        st.write('')
        st.write('Classificador 3: SVM \\\n \
                  Accuracy:  ' + str(round(model_metrics_json['XGBoost Accuracy'],2)) + '  \\\n \
                  Precision: ' + str(round(model_metrics_json['XGBoost Precision'],2)) + '  \\\n \
                  Recall:    ' + str(round(model_metrics_json['XGBoost Recall'],2)) + '  \\\n \
                  AUC score: ' + str(round(model_metrics_json['XGBoost AUC score'],2)) + '')
    with col2:
        st.write('Classificador 2: XGBost \\\n \
                  Accuracy:  ' + str(round(model_metrics_json['SVM Accuracy'],2)) + '  \\\n \
                  Precision: ' + str(round(model_metrics_json['SVM Precision'],2)) + '  \\\n \
                  Recall:    ' + str(round(model_metrics_json['SVM Recall'],2)) + '  \\\n \
                  AUC score: ' + str(round(model_metrics_json['SVM AUC score'],2)) + '')

        st.write('')
        st.write('')
        st.write('Classificador 4: Naive Bayes \\\n \
                  Accuracy:  ' + str(round(model_metrics_json['NB Accuracy'],2)) + '  \\\n \
                  Precision: ' + str(round(model_metrics_json['NB Precision'],2)) + '  \\\n \
                  Recall:    ' + str(round(model_metrics_json['NB Recall'],2)) + '  \\\n \
                  AUC score: ' + str(round(model_metrics_json['NB AUC score'],2)) + '')
   
        
    st.write('')
    st.write('Tenha em mente que nenhum algotírmo irá possuir uma taxa de acertos de 100%!')
st.write('Este é um projeto didático e seus resultados não devem ser utilizados como uma ferramenta de apoio para tomadas de decisões.')

st.caption('Alguns exemplos de notícias falsas: https://libguides.valenciacollege.edu/c.php?g=612299&p=4251645')
#st.caption('Alguns exemplos de notícias reais: https://www.politico.com/news/2022/11/29/donald-trump-campaign-00071225')
