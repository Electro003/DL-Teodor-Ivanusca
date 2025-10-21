import os

import pandas as pd
import numpy as np
import torch
from azure.ai.translation.text.models import InputTextItem
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from azure.core.credentials import AzureKeyCredential
from azure.ai.translation.text import TextTranslationClient
from dotenv import load_dotenv


load_dotenv()

data = pd.read_csv("CoQA_data.csv")
print("Number of question and answers: ", len(data))

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

random_num = np.random.randint(0,len(data))
question = data["question"][random_num]
text = data["text"][random_num]

input_ids = tokenizer.encode(question, text)
print("The input has a total of {} tokens.".format(len(input_ids)))

tokens = tokenizer.convert_ids_to_tokens(input_ids)

for token, id in zip(tokens, input_ids):
    print('{:8}{:8,}'.format(token,id))

sep_idx = input_ids.index(tokenizer.sep_token_id)
print("SEP token index: ", sep_idx)

num_seg_a = sep_idx+1
print("Number of tokens in segment A: ", num_seg_a)

num_seg_b = len(input_ids) - num_seg_a
print("Number of tokens in segment B: ", num_seg_b)

segment_ids = [0]*num_seg_a + [1]*num_seg_b

assert len(segment_ids) == len(input_ids)

output = model(torch.tensor([input_ids]),  token_type_ids=torch.tensor([segment_ids]))

answer_start = torch.argmax(output.start_logits)
answer_end = torch.argmax(output.end_logits)

if answer_end >= answer_start:
    answer = " ".join(tokens[answer_start:answer_end+1])
else:
    print("I am unable to find the answer to this question. Can you please ask another question?")


answer_start = torch.argmax(output.start_logits)
answer_end = torch.argmax(output.end_logits)

if answer_end >= answer_start:
    answer = tokens[answer_start]
    for i in range(answer_start+1, answer_end+1):
        if tokens[i][0:2] == "##":
            answer += tokens[i][2:]
        else:
            answer += " " + tokens[i]

if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."

key = os.environ["AZURE_DOCUMENT_TRANSLATION_KEY"]

credential = AzureKeyCredential(key)
text_translator = TextTranslationClient(credential=credential, region="eastus")

response = text_translator.translate(body = [answer], to_language = ['ro'])
translated_answer = response[0].translations[0].text if response else None


print("Question:{}".format(question.capitalize()))
print("Answer:{}.".format(answer.capitalize()))

print("Raspuns: {}.".format(translated_answer.capitalize()))