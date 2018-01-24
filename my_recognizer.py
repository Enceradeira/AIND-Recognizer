import warnings
from asl_data import SinglesData
import math


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    word_indices = range(len(test_set.wordlist))
    for probability, guess in map(lambda word_idx: recognize_word(word_idx, models, test_set), word_indices):
        probabilities.append(probability)
        guesses.append(guess)

    return probabilities, guesses


def recognize_word(word_idx, models, test_set):

    words_and_scores = dict(map(lambda m: score_word_with_model(word_idx, m, test_set), models.items()))

    max_log_likelihood = max(words_and_scores.values())
    best_guess = next(filter(lambda kv: kv[1] == max_log_likelihood, words_and_scores.items()))[0]

    return words_and_scores, best_guess


def score_word_with_model(word_idx, model, test_set):
    word_X, word_lengths = test_set.get_item_Xlengths(word_idx)
    try:
        score = model[1].score(word_X, word_lengths)
        return (model[0], score)
    except ValueError:
        return (model[0], -math.inf)
