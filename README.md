# Detección de Tweets de Desastres 

Un proyecto de Machine Learning y NLP para clasificar tweets y detectar si reportan desastres reales o no.

## Descripción
Este proyecto desarrolla modelos de clasificación para determinar automáticamente si un tweet está reportando un desastre real o no. Utiliza técnicas de procesamiento de lenguaje natural (NLP) y aprendizaje automático para distinguir entre tweets que informan sobre desastres reales y aquellos que usan lenguaje similar de manera figurativa.

## Objetivo
Construir y evaluar varios modelos de Machine Learning y Deep Learning para clasificar tweets en dos categorías:
- Desastre Real (1)
- No Desastre (0)

## Características Principales
- Análisis exploratorio de datos (EDA)
- Preprocesamiento profundo de texto
- Implementación de múltiples modelos:
  - Regresión Logística con TF-IDF
  - Regresión Logística con BoW
  - Red Neuronal LSTM

##  Resultados Principales
- Mejor modelo: Regresión Logística con TF-IDF
  - Accuracy: 82%
  - F1-Score: 0.78
- Comparativa de modelos:
  | Modelo | Accuracy | F1-Score |
  |--------|----------|-----------|
  | RL + TF-IDF | 82% | 0.78 |
  | RL + BoW | 81% | 0.77 |
  | LSTM | 80% | 0.75 |

## Tecnologías Utilizadas
- Python 3.x
- Principales bibliotecas:
  - pandas
  - numpy
  - scikit-learn
  - tensorflow
  - nltk
  - wordcloud
  - matplotlib
  - seaborn

##  Fases del Proyecto
1. **Análisis Exploratorio de Datos (EDA)**
   - Análisis de distribución de clases
   - Visualización de longitud de tweets
   - Nubes de palabras

2. **Preprocesamiento de Texto**
   - Limpieza de texto
   - Tokenización
   - Eliminación de stopwords
   - Lematización

3. **Modelado Baseline**
   - Implementación de vectorización TF-IDF y BoW
   - Regresión Logística

4. **Modelado Deep Learning**
   - Implementación de red LSTM
   - Comparación de resultados

## Licencia
[MIT](https://choosealicense.com/licenses/mit/)

##  Autor
Ricardo Goitia

## Enlaces
- [Dataset Original en Kaggle](https://www.kaggle.com/datasets/vbmokin/nlp-with-disaster-tweets-cleaning-data/data)

## Agradecimientos
- Kaggle por proporcionar el dataset
- Comunidad de NLP por recursos y herramientas
