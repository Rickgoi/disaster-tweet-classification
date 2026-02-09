import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_tweet(text):
    """
    Función profesional para limpiar tweets:
    - Convierte a minúsculas.
    - Elimina URLs, menciones (@) y hashtags (#).
    - Elimina puntuación y stopwords.
    """
    text = str(text).lower()
    # Eliminar URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Eliminar menciones y hashtags
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'#', '', text)
    # Eliminar puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Eliminar stopwords y palabras cortas
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text
