import re
import numpy as np
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from collections import Counter

class FullTextSearch:
    """
        BM25 algorithm
    """

    def __init__(self, documents, k1, b):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.N = len(documents)
        self.tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        self.train_tokenizer()
        self.token_frequency = self.count_frequency_tokens(documents)
        self.tokens_by_document = self.count_tokens_by_document(documents)
        self.avg_length = sum([t.total() for t in self.tokens_by_document]) / self.N

    def train_tokenizer(self):
        """Тренировка словаря-токенайзера"""
        self.tokenizer.normalizer = normalizers.Sequence(
            [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        )
        self.tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        special_tokens = ["[UNK]"]
        trainer = trainers.WordPieceTrainer(vocab_size=15000, special_tokens=special_tokens)
        self.tokenizer.train_from_iterator(self.get_training_corpus(), trainer=trainer)
        self.tokenizer.decoder = decoders.WordPiece(prefix="##")
        self.tokenizer.save("bm25tokenizer.json")

    def get_training_corpus(self):
        """Генерация корпуса для словаря-токенайзера"""
        for i in range(0, len(self.documents), 1000):
            yield self.documents[i: i + 1000]

    def count_frequency_tokens(self, documents):
        """Подсчет частоты тоекнов в документах"""
        counts = {}
        for doc in documents:
            for token_id in set(self.tokenizer.encode(doc).ids):
                counts[token_id] = counts.get(token_id, 0) + 1
        return counts

    def count_tokens_by_document(self, documents):
        """Подсчет токенов в документе"""
        counts = [Counter(self.tokenizer.encode(doc).ids) for doc in documents]
        return counts

    def idf(self, token_ids):
        """Подсчет IDF-токенов"""
        # Исправление: обработка неизвестных токенов
        counts = []
        for token_id in token_ids:
            freq = self.token_frequency.get(token_id, 0) # возвращает 0, если токена нет
            counts.append(freq)
        counts = np.array(counts)
        return np.log((self.N - counts + 0.5) / (counts + 0.5) + 1)

    def token_frequency_by_document(self, token_ids):
        """Подсчет частоты токенов в документе"""
        result = []
        for tokens in self.tokens_by_document:
            doc_length = tokens.total()
            # Исправление: обработка неизвестных токенов
            counts = []
            for token_id in token_ids:
                freq = tokens.get(token_id, 0) # возвращает 0, если токена нет в документе
                counts.append(freq)
            counts = np.array(counts)
            delimiter = counts + self.k1 * (1 - self.b + self.b * doc_length / self.avg_length)
            result.append(counts * (self.k1 + 1) / delimiter)
        return np.array(result)

    def get_scores(self, query):
        """Получение баллов BM25 для запроса"""
        token_ids = self.tokenizer.encode(query).ids
        # remove [UNK] tokens (id == 0)
        token_ids = [tid for tid in token_ids if tid != 0]

        # Проверка: если после фильтрации нет токенов
        if not token_ids:
            return np.zeros(self.N)

        token_freq = self.token_frequency_by_document(token_ids)
        idf = self.idf(token_ids)
        # BM25 score calculation
        # score = sum( (tf / (tf + k1)) * idf )
        # token_freq уже содержит (tf * (k1 + 1)) / (tf + denominator)
        # где denominator = k1 * (1 - b + b * doc_length / avg_length)
        # Поэтому score = sum( (tf * (k1 + 1)) / (tf + denominator) * idf )
        # Или score = sum( (tf * (k1 + 1) / denominator) * idf / (k1 + 1) ) = sum( (tf / denominator) * idf )
        # token_freq = (tf * (k1 + 1)) / denominator
        # Поэтому score = sum( token_freq * idf )
        result = token_freq @ idf # Matrix multiplication: (N_docs x N_tokens) @ (N_tokens,) -> (N_docs,)
        return result
        # print(len(token_ids), token_freq.shape, idf.shape)
        # print(result.shape)

def main():
    pass

if __name__ == '__main__':
    main()