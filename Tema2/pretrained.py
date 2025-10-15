import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def predict_next_n_words(prompt_text, n=2):
    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Please ensure you have an active internet connection to download the model weights.")
        return ""

    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
    current_length = input_ids.shape[1]
    max_gen_length = current_length + n

    print("-" * 50)
    print(f"Input prompt (tokens): {tokenizer.tokenize(prompt_text)}")
    print(f"Current token length: {current_length}")
    print(f"Target prediction length: {n} words (tokens)")
    print("-" * 50)

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_gen_length,
        num_beams=1,  # Using 1 beam search is faster for simple prediction
        do_sample=False,  # Use greedy decoding (highest probability token at each step)
        pad_token_id=tokenizer.eos_token_id  # Set padding to End-of-Sequence token
    )

    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

    predicted_text_with_context = tokenizer.decode(generated_sequence, skip_special_tokens=True)
    prompt_ids = input_ids[0].tolist()
    generated_ids = output_sequences[0].tolist()
    predicted_tokens = generated_ids[len(prompt_ids):]
    predicted_words = tokenizer.decode(predicted_tokens, clean_up_tokenization_spaces=True)

    return predicted_words.strip()


input_sequence = "The quick brown fox"

words_to_predict = 2

prediction = predict_next_n_words(input_sequence, words_to_predict)

print(f"Input Sequence: '{input_sequence}'")
print(f"Predicted Next {words_to_predict} Words: '{prediction}'")
print(f"Full Generated Sequence: '{input_sequence} {prediction}'")
