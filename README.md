# Detección de Tweets de Desastres

## 1. Descripción del Problema de Negocio

En la era de la información instantánea, agencias de noticias, servicios de emergencia y organizaciones humanitarias dependen del monitoreo de redes sociales como Twitter para detectar eventos importantes en tiempo real. Sin embargo, el lenguaje humano es inherentemente ambiguo. Un tweet que dice "¡Se incendió la casa!" puede ser un reporte de un desastre real o una expresión metafórica.

Existe una necesidad crítica de un sistema automatizado que pueda filtrar rápidamente el "ruido" y distinguir los tweets que anuncian un desastre real de aquellos que no. Esto permitiría una respuesta más rápida y eficiente ante emergencias, salvando potencialmente vidas y recursos.

## 2. Objetivo General

Construir y evaluar varios modelos de Machine Learning y Deep Learning para clasificar tweets en dos categorías: Desastre Real (1) o No Desastre (0), basándose únicamente en el contenido textual del tweet.

## 3. Origen de los Datos

Los datos provienen de la competencia de Kaggle "Real or Not? NLP with Disaster Tweets", que a su vez se basan en el notebook público "NLP with Disaster Tweets - EDA and Cleaning data".

**Links:**
*   [Kaggle Dataset](https://www.kaggle.com/datasets/vbmokin/nlp-with-disaster-tweets-cleaning-data/data?select=train_data_cleaning2.csv)
*   [Kaggle Notebook de Origen](https://www.kaggle.com/code/vbmokin/nlp-with-disaster-tweets-eda-and-cleaning-data#5.-Save-and-visualization-of-cleaning-datasets)

## 4. Definición de las Variables

*   **id**: Un identificador único para cada tweet.
*   **keyword**: Una palabra clave específica del tweet (puede estar en blanco).
*   **location**: La ubicación desde donde se envió el tweet (puede estar en blanco y es poco consistente).
*   **text**: Contenido textual del tweet (variable predictora principal).
*   **target**: Variable objetivo a predecir:
    *   **1**: El tweet corresponde a un desastre real.
    *   **0**: El tweet no corresponde a un desastre real.

## 5. Metodología

El proyecto se estructuró en varias fases:

### Fase 1: Análisis Exploratorio de Datos (EDA) y Preprocesamiento Inicial

*   **Inspección Básica**: Se revisó la información general del dataset, las primeras filas, el conteo de valores nulos y la distribución de la variable `target`. Se observó un ligero desbalance de clases (4342 no-desastre vs. 3271 desastre).
*   **Análisis de Longitud de Tweets**: Se creó una columna `text_length` para analizar la distribución de la longitud de los tweets por clase. Se encontró que los tweets de no-desastre tienden a ser ligeramente más largos (pico entre 120-140 caracteres) que los de desastre (pico alrededor de 100 caracteres).
*   **Nubes de Palabras (Pre-limpieza)**: Visualizaciones iniciales mostraron palabras clave relevantes en cada categoría, así como la presencia de "ruido" como URLs (`http co`).

### Fase 2: Preprocesamiento Profundo del Texto

Se aplicó una función `clean_text` para:
1.  Eliminar URLs, etiquetas HTML, menciones (`@usuario`) y hashtags (`#`).
2.  Eliminar caracteres no alfabéticos (puntuación, números).
3.  Convertir todo el texto a minúsculas.
4.  Tokenizar el texto.
5.  Eliminar stopwords (palabras comunes).
6.  Aplicar lematización para reducir palabras a su forma base.
*   **Nubes de Palabras (Post-limpieza)**: Se generaron nuevas nubes de palabras que confirmaron una limpieza efectiva y resaltaron términos específicos de desastres (fire, flood, suicide, bomber) y no-desastres (love, like, new, day).
*   **Análisis de N-gramas**: Se identificaron los bigramas más comunes. En tweets de desastre, predominaron combinaciones como "malaysia airline", "suicide bomber", "oil spill". En tweets de no-desastre, destacaron "body bag" (uso metafórico), "youtube video", "cross body".
*   **Análisis de Sentimiento**: Utilizando `TextBlob`, se calculó la polaridad del sentimiento. Se encontró que un sentimiento muy positivo es un fuerte indicador de tweets de no-desastre, mientras que sentimientos neutros o negativos se distribuyen en ambas clases.

### Fase 3: Vectorización y Modelo Baseline (Machine Learning Clásico)

Se utilizaron dos técnicas de vectorización con un modelo de Regresión Logística:
*   **TF-IDF (Term Frequency-Inverse Document Frequency)**: Da más importancia a las palabras relevantes pero no comunes.
*   **Bag of Words (BoW)**: Conteo simple de la frecuencia de palabras.

**Resultados del Baseline:**

*   **Regresión Logística + TF-IDF**:
    *   Accuracy: 82%
    *   F1-Score (clase 1 - Desastre): 0.78
    *   Falsos Negativos (error crítico): 181

*   **Regresión Logística + BoW**:
    *   Accuracy: 81%
    *   F1-Score (clase 1 - Desastre): 0.77
    *   Falsos Negativos: 177

**Conclusión de la Fase 3**: TF-IDF fue seleccionado como el baseline oficial debido a su mejor equilibrio entre precisión y recall (F1-score superior), a pesar de que BoW tuvo ligeramente menos Falsos Negativos, lo hizo a costa de más Falsos Positivos.

### Fase 4: Modelado con Deep Learning

Se construyó un modelo de Red Neuronal Recurrente (LSTM) utilizando Keras (TensorFlow):
1.  **Tokenización y Secuenciamiento**: Conversión de texto a secuencias numéricas.
2.  **Padding**: Estandarización de la longitud de las secuencias.
3.  **Arquitectura LSTM**: Capa de Embedding, `SpatialDropout1D`, `LSTM` con `dropout` y `recurrent_dropout`, y `Dense` con activación `sigmoid`.
4.  **Entrenamiento**: Se utilizó `EarlyStopping` para prevenir el sobreajuste.

**Resultados del Modelo LSTM**:

*   Accuracy: 81%
*   F1-Score (clase 1 - Desastre): 0.77
*   Falsos Negativos: 195

## 6. Conclusiones Finales

*   **TF-IDF superó a BoW**: La ponderación de TF-IDF fue más efectiva que el simple conteo de palabras para este problema.
*   **El modelo clásico superó al LSTM**: Sorprendentemente, el modelo de Regresión Logística con TF-IDF obtuvo un rendimiento ligeramente superior al modelo LSTM. El LSTM tuvo más Falsos Negativos, lo cual es el error más crítico en este contexto. Esto podría deberse al tamaño del dataset, la simplicidad del modelo LSTM implementado o la necesidad de un ajuste de hiperparámetros más exhaustivo para el DL.

## 7. Recomendación de Negocio

Se recomienda implementar el **modelo de Regresión Logística con TF-IDF**. Ofrece el mejor equilibrio entre rendimiento (mayor F1-score) y eficiencia. Es un modelo más simple, más rápido de entrenar y más fácil de interpretar, y demostró ser más preciso para la detección de desastres en este caso específico.

## 8. Perspectivas Futuras del Proyecto

1.  **Ingeniería de Características Adicional**: Incorporar características como la presencia de palabras clave específicas, el número de menciones, o la polaridad del sentimiento como inputs adicionales al modelo.
2.  **Ajuste de Hiperparámetros**: Realizar una búsqueda sistemática de los mejores hiperparámetros para `TfidfVectorizer` (ej. `ngram_range`, `min_df`) y para el modelo LSTM (ej. `output_dim` del embedding, número de unidades LSTM, etc.).
3.  **Modelos de Deep Learning más Avanzados (Transfer Learning)**: La mejora más prometedora sería utilizar un modelo de Transformer pre-entrenado como BERT (o una versión más ligera como DistilBERT). Estos modelos suelen ofrecer resultados de mayor calidad en tareas de clasificación de texto mediante fine-tuning.

## 9. Requisitos del Proyecto (requirements.txt)

```
pandas==2.2.2
numpy==2.0.2
matplotlib==3.10.0
seaborn==0.13.2
wordcloud==1.9.4
nltk==3.9.1
scikit-learn==1.6.1
tensorflow==2.19.0
textblob==0.19.0
```
