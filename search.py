import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import pymorphy2
from tqdm import tqdm
import time
import joblib

# загрузка необходимых ресурсов для обработки текста

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# инициализация морфологического анализатора и списка стоп-слов для русского
morph = pymorphy2.MorphAnalyzer()
russian_stopwords = set(stopwords.words('russian'))

def preprocess_text(text):
    """
    Функция для предобработки текста:
    1. Приведение к нижнему регистру.
    2. Токенизация.
    3. Нормализация слов с использованием морфологического анализатора.
    4. Удаление стоп-слов.

    :param text: Строка текста для обработки.
    :return: Предобработанный текст.
    """
    print(f"Предобработка текста: {text}")
    words = nltk.word_tokenize(text.lower())
    words = [morph.parse(word)[0].normal_form for word in words if word.isalnum()]
    words = [word for word in words if word not in russian_stopwords]
    return ' '.join(words)

def full_text_search(query, top_k=5):
    """
    Полнотекстовый поиск с использованием TF-IDF.

    :param query: Поисковый запрос.
    :param top_k: Количество возвращаемых результатов.
    :return: Список кортежей (заголовок, оценка сходства).
    """
    print("Выполняется полнотекстовый поиск...")
    processed_query = preprocess_text(query)
    query_vec = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    print("Полнотекстовый поиск завершен.")
    return [(data.iloc[i]['title'], similarities[i]) for i in top_indices]

def get_embedding(text):
    """
    Получение эмбеддинга текста с использованием модели sberbank-ai/sbert_large_nlu_ru.

    :param text: Текст для преобразования в эмбеддинг.
    :return: Эмбеддинг текста в виде numpy массива.
    """
    print(f"Получение эмбеддинга для: {text}")
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=512
    )
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    print("Эмбеддинг получен.")
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def vector_search(query, top_k=5):
    """
    Векторный поиск с использованием эмбеддингов.

    :param query: Поисковый запрос.
    :param top_k: Количество возвращаемых результатов.
    :return: Список кортежей (заголовок, оценка сходства).
    """
    print("Выполняется векторный поиск...")
    query_vec = get_embedding(query)
    similarities = cosine_similarity([query_vec], sbert_embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    print("Векторный поиск завершен.")
    return [(data.iloc[i]['title'], similarities[i]) for i in top_indices]

def hybrid_search(query, top_k=5, weight_full_text=0.5):
    """
    Гибридный поиск, комбинирующий полнотекстовый и векторный подходы.

    :param query: Поисковый запрос.
    :param top_k: Количество возвращаемых результатов.
    :param weight_full_text: Вес для полнотекстового поиска в гибридной модели.
    :return: Список кортежей (заголовок, комбинированная оценка сходства).
    """
    print("Выполняется гибридный поиск...")
    full_text_results = full_text_search(query, top_k)
    vector_results = vector_search(query, top_k)

    combined_results = {}
    for doc, score in full_text_results:
        combined_results[doc] = weight_full_text * score

    for doc, score in vector_results:
        if doc in combined_results:
            combined_results[doc] += (1 - weight_full_text) * score
        else:
            combined_results[doc] = (1 - weight_full_text) * score

    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
    print("Гибридный поиск завершен.")
    return sorted_results[:top_k]

def perform_search(query, top_k, method):
    """
    Основная функция для выполнения поиска по выбранному методу.

    :param query: Поисковый запрос.
    :param top_k: Количество возвращаемых результатов.
    :param method: Метод поиска ('full_text', 'vector', 'hybrid').
    """
    start_time = time.time()

    if method == 'full_text':
        results = full_text_search(query, top_k)
    elif method == 'vector':
        results = vector_search(query, top_k)
    elif method == 'hybrid':
        results = hybrid_search(query, top_k)
    else:
        raise ValueError("Неверный метод поиска. Выберите из 'full_text', 'vector', 'hybrid'.")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # нумерация результатов и вывод
    numbered_results = [(i + 1, doc, score) for i, (doc, score) in enumerate(results)]

    print(f"\nВаш поисковой запрос: {query}")
    print(f"\nВыбранный метод поиска: {method}")
    print(f"\nКоличество результатов для выдачи: {top_k}")
    print(f"Время исполнения: {elapsed_time:.4f} секунд")
    for i, doc, score in numbered_results:
        print(f"{i}. Скор: {score:.4f}\nЗаглавие: {doc}\n")

# Загрузка данных и моделей
print("Загрузка данных и моделей...")
data = pd.read_csv('rus_novel_titles.csv')

# Загрузка заранее обученных моделей и данных
tfidf_matrix = joblib.load('tfidf_matrix.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
sbert_embeddings = np.load('sbert_large_nlu_ru_embeddings.npy')

# Инициализация токенизатора и модели для векторного поиска
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

# GPU, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Данные и модели загружены.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Поиск')
    parser.add_argument('--query', type=str, required=True, help='Поисковый запрос')
    parser.add_argument('--count', type=int, default=5, help='Количество возвращаемых результатов')
    parser.add_argument('--method', type=str, default='hybrid', choices=['full_text', 'vector', 'hybrid'], help='Метод поиска')
    args = parser.parse_args()

    perform_search(args.query, args.count, args.method)
