# 🚨 Real-Time Disaster Detection in Social Media (NLP & Deep Learning)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) 
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![NLP](https://img.shields.io/badge/NLP-Sentiment--Analysis-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Resumen Ejecutivo
En situaciones de crisis, la velocidad de la información salva vidas. Este proyecto desarrolla un sistema de **Procesamiento de Lenguaje Natural (NLP)** capaz de clasificar tweets para detectar desastres reales en tiempo real, distinguiendo reportes verídicos de usos metafóricos del lenguaje (ej: *"Esa fiesta fue una bomba"* vs *"Explosión en el centro"*).

Se implementó un pipeline completo de Ciencia de Datos, comparando modelos estadísticos clásicos (**TF-IDF**) con arquitecturas de **Deep Learning (LSTM Bidireccional)**.

## 🛠️ Tech Stack & Herramientas
* **Lenguaje:** Python.
* **NLP & Preprocesamiento:** `NLTK`, `TextBlob`, `Regex`, `Tokenization`.
* **Machine Learning:** `Scikit-Learn`, `XGBoost`, `TF-IDF`.
* **Deep Learning:** `TensorFlow/Keras`, `LSTM Bidireccional`, `Embeddings`.
* **Visualización:** `Seaborn`, `Matplotlib`, `WordClouds`.

## 🏗️ Estructura del Proyecto
El proyecto está organizado siguiendo estándares de ingeniería de software para asegurar la reproducibilidad:
* `src/text_processor.py`: Script modular de preprocesamiento de texto (refactorizado para producción).
* `notebooks/Deteccion_de_tweets_desastres.ipynb`: Pipeline completo de EDA, entrenamiento y evaluación.
* `models/`: Modelos entrenados y serializados (.h5 / .pkl).

## 📊 Hallazgos y Resultados (Benchmark)

Tras la experimentación, se evaluaron dos enfoques principales:

| Modelo | Accuracy | Precisión (Desastres) | F1-Score | Ventaja Estratégica |
| :--- | :---: | :---: | :---: | :--- |
| **TF-IDF + Modelo Clásico** | **83%** | **86%** | **0.79** | Alta velocidad y precisión. |
| **Deep Learning (LSTM)** | 0.80 | 0.83 | 0.75 | Captura contexto secuencial complejo. |

### 💡 Key Insights:
1.  **Filtrado de Ruido:** El modelo de Deep Learning alcanzó un **Recall de 0.90** en la clase "No Desastre", lo que garantiza que las agencias de emergencia no sean saturadas con falsas alarmas.
2.  **Neutralidad Informativa:** El análisis de sentimiento reveló que los desastres reales mantienen una polaridad neutra (tono informativo), mientras que la alta subjetividad suele indicar contenido irrelevante para emergencias.
3.  **Contexto Dual:** La implementación de **LSTM Bidireccional** permitió captar la semántica de mensajes cortos donde el orden de las palabras altera drásticamente el significado.

## 🚀 Instalación y Uso
1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/TuUsuario/disaster-tweets-nlp.git](https://github.com/TuUsuario/disaster-tweets-nlp.git)
    ```
2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Procesar datos de forma modular:**
    ```python
    from src.text_processor import clean_tweet
    tweet_limpio = clean_tweet("🚨 Fuego detectado en el sector 4! #Emergencia")
    ```

## 🎯 Impacto de Negocio
Este sistema reduce el tiempo de monitoreo manual en redes sociales, permitiendo a las organizaciones humanitarias y servicios de emergencia priorizar recursos basados en alertas con un **86% de confiabilidad**.

---
📫 **Contacto:** (https://www.linkedin.com/in/ricardo-goitia-659a5895/)
