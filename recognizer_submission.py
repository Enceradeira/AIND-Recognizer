from my_recognizer import recognize
from asl_utils import show_errors
from my_model_selectors import *
import numpy as np
import pandas as pd
from asl_data import AslDb
from matplotlib import (cm, pyplot as plt, mlab)
from asl_data import SinglesData
from itertools import product

asl = AslDb()  # initializes the database

### Ground Features
features_ground = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

### Polar Features
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']


def calculate_polar_radius(x, y):
    return (asl.df[x].pow(2) + asl.df[y].pow(2)).pow(0.5)


def calculate_polar_theta(x, y):
    return np.arctan2(asl.df[x], asl.df[y])


asl.df[features_polar[0]] = calculate_polar_radius('grnd-rx', 'grnd-ry')
asl.df[features_polar[1]] = calculate_polar_theta('grnd-rx', 'grnd-ry')
asl.df[features_polar[2]] = calculate_polar_radius('grnd-lx', 'grnd-ly')
asl.df[features_polar[3]] = calculate_polar_theta('grnd-lx', 'grnd-ly')

### Delta Features
features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']


def calculate_diff(column):
    return asl.df[column].diff().fillna(0)


asl.df[features_delta[0]] = calculate_diff('right-x')
asl.df[features_delta[1]] = calculate_diff('right-y')
asl.df[features_delta[2]] = calculate_diff('left-x')
asl.df[features_delta[3]] = calculate_diff('left-y')

### Norm Features
features_norm = ['norm-rx', 'norm-ry', 'norm-lx', 'norm-ly']
df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()


def calculate_norm(coordinate):
    speaker = asl.df['speaker']
    return (asl.df[coordinate] - (speaker.map(df_means[coordinate]))) / (speaker.map(df_std[coordinate]))


asl.df[features_norm[0]] = calculate_norm('right-x')
asl.df[features_norm[1]] = calculate_norm('right-y')
asl.df[features_norm[2]] = calculate_norm('left-x')
asl.df[features_norm[3]] = calculate_norm('left-y')


## Ground Norm-Feature
features_ground_norm = ['grnd-norm-rx', 'grnd-norm-ry', 'grnd-norm-lx', 'grnd-norm-ly']
asl.df[features_ground_norm[0]] = calculate_norm(features_ground[0])
asl.df[features_ground_norm[1]] = calculate_norm(features_ground[1])
asl.df[features_ground_norm[2]] = calculate_norm(features_ground[2])
asl.df[features_ground_norm[3]] = calculate_norm(features_ground[3])

## Polar Norm-Feature
features_polar_norm = ['polar-norm-rr', features_polar[1], 'polar-norm-lr', features_polar[3]]
asl.df[features_polar_norm[0]] = calculate_norm(features_polar[0])
asl.df[features_polar_norm[2]] = calculate_norm(features_polar[2])


def calculate_error_rate(guesses: list, test_set: SinglesData):
    """ Print WER and sentence differences in tabular form

    :param guesses: list of test item answers, ordered
    :param test_set: SinglesData object
    :return:
        nothing returned, prints error report

    WER = (S+I+D)/N  but we have no insertions or deletions for isolated words so WER = S/N
    """
    S = 0
    N = len(test_set.wordlist)
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        print("Size of guesses must equal number of test words ({})!".format(num_test_words))
    for word_id in range(num_test_words):
        if guesses[word_id] != test_set.wordlist[word_id]:
            S += 1

    return float(S) / float(N)


def train_all_words(features, model_selector):
    training = asl.build_training(features)  # Experiment here with different feature sets defined in part 1
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                               n_constant=3).select()
        model_dict[word] = model
    return model_dict


# TODO Choose a feature set and model selector
def recognize_with(features, model_selector):
    models = train_all_words(features, model_selector)
    test_set = asl.build_test(features)
    probabilities, guesses = recognize(models, test_set)
    return guesses, test_set


def recognize_with_combination(selector_and_features):
    features = selector_and_features[1]
    selector = selector_and_features[0]
    guesses, test_set = recognize_with(features[1], selector[1])
    return calculate_error_rate(guesses, test_set)


def print_to_grid(results):
    [print(f"{result[0][0]}\t|\t{result[1][0]}\t|\t{result[2]}") for result in results]

selectors = {"cv": SelectorCV, "bic": SelectorBIC, "dic": SelectorDIC}
features = {"ground": features_ground, "norm": features_norm, "delta": features_delta, "polar": features_polar, "grnd-norm": features_ground_norm, "polar-norm": features_polar_norm}
nr_runs = len(selectors) * len(features)

combinations = [(s, f) for s in selectors.items() for f in features.items()]
results = map(lambda c: (c[0], c[1], recognize_with_combination(c)), combinations)

print("\n\nBEGIN Result")
print_to_grid(results)
print("\n\nEND Result")
