import nltk
import spacy

grammar_string = """
    S -> NP VP

    NP -> VBG NNS
    NP -> JJ NNS
    NP -> Det NN
    NP -> Det NNS PP
    NP -> NP CC NP
    NP -> NP PP

    VP -> MD VB JJ
    VP -> VBD VBG
    VP -> VBZ NP
    VP -> VP PP

    PP -> IN NP

    # Lexicon
    Det -> 'The' | 'the'
    NN -> 'bride' | 'groom'
    NNS -> 'planes' | 'parents'
    JJ -> 'dangerous' | 'Flying'
    VBG -> 'flying'
    VBZ -> 'loves'
    VBD -> 'were'
    MD -> 'can'
    VB -> 'be'
    IN -> 'of' | 'more_than'
    CC -> 'and'
"""


my_grammar = nltk.CFG.fromstring(grammar_string)
parser = nltk.ChartParser(my_grammar)

sent1_tokens = ['Flying', 'planes', 'can', 'be', 'dangerous']
sent2_tokens = ['The', 'parents', 'of', 'the', 'bride', 'and', 'the', 'groom', 'were', 'flying']
sent3_tokens = ['The', 'groom', 'loves', 'dangerous', 'planes', 'more_than', 'the', 'bride']

for tree in parser.parse(sent1_tokens):
    tree.pretty_print()

for tree in parser.parse(sent2_tokens):
    tree.pretty_print()

for tree in parser.parse(sent3_tokens):
    tree.pretty_print()

text = ("Flying planes can be dangerous."
        "The parents of the bride and the groom were flying."
        "The groom loves dangerous planes more_than the bride.")

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
for token in doc:
        print(f"{token.text:<12} | Head: {token.head.text:<12} | Dep: {token.dep_:<12} | POS: {token.pos_}")

deps = [(token.text, token.dep_, token.head.text) for token in doc]
print(deps)

### An application where syntactic and/or dependency parsing are needed is a grammar checking tool.
### Such a tool can analyze the structure of sentences to identify grammatical errors and suggest corrections.
### By understanding the relationships between words and phrases, the tool can provide more accurate feedback on sentence construction,
### helping users improve their writing skills. This is a particularly useful for non-native speakers because we all know
### that translating directly from one language to another can lead to awkward or incorrect phrasing. By leveraging syntactic and dependency parsing,
### we can create a more sophisticated grammar checking tool that goes beyond simple spell-checking and addresses deeper linguistic issues.
