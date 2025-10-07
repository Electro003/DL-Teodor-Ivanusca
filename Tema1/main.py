import random
from nltk.corpus import wordnet as wn

def get_random_word_and_synsets():
    all_lemmas = list(wn.all_lemma_names(pos='n'))
    rand_word = random.choice(all_lemmas)
    word_synsets = wn.synsets(rand_word)

    return rand_word, word_synsets[0]



def check_if_user_word_in_wordnet(user_word):
    synsets = wn.synsets(user_word)
    intended = []
    for syn in synsets:
        if syn.name().split('.')[0] == user_word:
            intended.append(syn)

    return bool(intended), intended


def main():
    points = 0
    fails = 0
    system_word, system_word_synset = get_random_word_and_synsets()
    print("The random concept is: " + system_word.replace("_", " "))
    while fails < 3:
        user_word = input("Enter a similar concept: ")
        is_in_wordnet, user_word_synsets = check_if_user_word_in_wordnet(user_word.lower().replace(" ", "_"))
        if is_in_wordnet:
            max_distance = 0
            for user_syn in user_word_synsets:
                wup_sim = system_word_synset.wup_similarity(user_syn)
                if wup_sim and wup_sim > max_distance:
                    max_distance = wup_sim

            if max_distance > 0.5:
                points += int(max_distance * 10)
                print(f"You scored {int(max_distance * 10)} points! Total points: {points}")

                if max_distance > 0.5:
                    print("Good! You found a similar concept!")

                if max_distance > 0.8:
                    print("Awesome! You found a very similar concept!")

                if max_distance > 0.9:
                    print("Great! You nailed the colocation!")
            else:
                fails += 1
                print("The concepts are not similar enough. Try again.")

        else:
            print("The word is not in WordNet. Please try again.")
            continue

if __name__ == "__main__":
    main()



