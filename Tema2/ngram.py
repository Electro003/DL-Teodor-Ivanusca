from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from collections import defaultdict


class NGram:
    def __init__(self, corpus, n):
        self.n = n
        self.corpus = [word.lower() for text in corpus for word in word_tokenize(text)]
        self.vocab = set(self.corpus)
        self.vocab_size = len(self.vocab)

        self.ngrams_count = self.compute_n_grams_count(self.n)
        self.n_minus_1_grams_count = self.compute_n_grams_count(self.n - 1)

    def compute_n_grams_count(self, gram_n):
        if gram_n <= 0:
            return defaultdict(int)

        counts = defaultdict(int)
        for i in range(len(self.corpus) - gram_n + 1):
            n_gram = tuple(self.corpus[i:i + gram_n])
            counts[n_gram] += 1
        return counts

    def predicts(self, context):
        tokenized_input = word_tokenize(context.lower())

        if len(tokenized_input) < self.n - 1:
            print(f"Context is too short for a {self.n}-gram model.")
            return []

        last_ngram_context = tuple(tokenized_input[-(self.n - 1):])

        vocab_probs = {}
        for word in self.vocab:
            full_n_gram = last_ngram_context + (word,)

            numerator = self.ngrams_count.get(full_n_gram, 0) + 1
            denominator = self.n_minus_1_grams_count.get(last_ngram_context, 0) + self.vocab_size

            probability = numerator / denominator
            vocab_probs[word] = probability

        top_suggestions = sorted(vocab_probs.items(), key=lambda x: x[1], reverse=True)[:self.n]
        return top_suggestions


romaninan_corpus = [
    "Implică-te! Vino și află mai multe joi, 15 octombrie, ora 20:00 în sala CH1 din Facultatea de Inginerie Chimică și Protecția Mediului. Până atunci urmărește-ne pe www.bestis.ro ! Vezi cele mai tari Restaurante Nunti in Iasi, click pentru detalii",
    "București, capitala, este un oraș vibrant, plin de istorie și modernitate. Aici, vizitatorii pot explora Palatul Parlamentului, a doua cea mai mare clădire administrativă din lume, și pot admira arhitectura eclectică a clădirilor din centrul vechi.",
    "În Transilvania, castelele medievale, cum ar fi Castelul Bran, cunoscut și sub numele de Castelul lui Dracula, atrag mii de turiști în fiecare an. Satele săsești fortificate, precum Viscri, păstrează o atmosferă autentică și sunt incluse în patrimoniul UNESCO.",
    "Tradițiile românești sunt adânc înrădăcinate în cultura țării. Sărbătorile, precum Mărțișorul sau Colindatul, sunt momente de bucurie și comuniune. Oamenii poartă costume populare tradiționale și participă la festivaluri de muzică și dans.",
    "Gastronomia românească este delicioasă și variată. Mâncăruri tradiționale precum sarmale, mămăligă și mici sunt extrem de populare. Vinurile românești au câștigat recunoaștere internațională, iar podgoriile din țară sunt o destinație populară pentru iubitorii de vin.",
    "Educația în România a cunoscut o serie de transformări în ultimii ani. Elevii și studenții români participă la competiții internaționale și obțin rezultate remarcabile. Sistemul universitar oferă o gamă largă de specializări, iar multe universități sunt recunoscute la nivel european.",
    "Tehnologia și inovația au un rol tot mai important în economia României. Industria IT este în plină expansiune, iar start-up-urile locale atrag investiții semnificative. Există un interes crescut pentru dezvoltarea de soluții software, inteligență artificială și securitate cibernetică.",
    "Cultura românească este un amestec de influențe diverse, de la cele balcanice și slave până la cele latine și occidentale. Marii scriitori, poeți și artiști români au lăsat o moștenire culturală de neprețuit, iar teatrele și muzeele din țară sunt o dovadă a bogăției artistice.",
    "Sportul este o altă pasiune a românilor. Fotbalul este cel mai popular sport, dar și alte discipline, precum gimnastica, tenisul și handbalul, au adus medalii și prestigiu țării. Jucătorii români sunt apreciați la nivel internațional pentru talentul și performanța lor."
]
ngram_model = NGram(corpus=romaninan_corpus, n=3)

context = "cunoscut o serie "
predictions = ngram_model.predicts(context)
print(f"'{context}' (n={ngram_model.n}):")
print(predictions)