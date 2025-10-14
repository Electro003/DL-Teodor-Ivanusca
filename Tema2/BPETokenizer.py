from transformers import AutoTokenizer
from collections import defaultdict

class BPETokenizer:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.word_freq = defaultdict(int)
        self.compute_word_freq()
        self.alphabet = []
        self.compute_alphabet()
        self.vocab = ["<|endoftext|>"] + self.alphabet.copy()
        self.splits = {word: [c for c in word ] for word in self.word_freq.keys()}
        self.merges = {}

    def compute_word_freq(self):
        for text in self.corpus:
            words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                self.word_freq[word] += 1

    def compute_alphabet(self):
        for word in self.word_freq.keys():
            for char in word:
                if char not in self.alphabet:
                    self.alphabet.append(char)
        self.alphabet.sort()

    def compute_pair_freq(self, splits):
        pair_freq = defaultdict(int)
        for word, freq in self.word_freq.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split)-1):
                pair = (split[i], split[i+1])
                pair_freq[pair] += freq

        return pair_freq

    def merge_pair(self, a, b):
        for word in self.word_freq:
            split = self.splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i+1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1

            self.splits[word] = split

    def train(self, n):
        while len(self.vocab) < n:
            pair_freq = self.compute_pair_freq(self.splits)

            best_pair = ""
            max_freq = None

            for pair, freq in pair_freq.items():
                if max_freq is None or max_freq < freq:
                    max_freq = freq
                    best_pair = pair

            self.merge_pair(*best_pair)
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.append(best_pair[0] + best_pair[1])

    def tokenize(self, text):
        pre_tokenize_result = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        splits = [[l for l in word] for word in pre_tokenized_text]

        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i+1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1

                splits[idx] = split

        return sum(splits, [])

if __name__ == '__main__':
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    my_tokenizer =  BPETokenizer(corpus)
    my_tokenizer.train(50)
    print(my_tokenizer.tokenize("This is not a token."))

