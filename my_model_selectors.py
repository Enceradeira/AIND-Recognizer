import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
from abc import ABC, abstractmethod


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        return self.create_and_fit_model(num_states, self.X, self.lengths)

    def create_and_fit_model(self, num_states, x, lengths):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(x, lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class ModelSelectorUsingCV(ModelSelector, ABC):
    ''' selects best model by cross-validating folds that are based on given word sequences

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        trials = range(self.min_n_components, self.max_n_components)
        models_and_scores = list(map(self.scoreTrial, trials))
        best_model_and_score = max(models_and_scores, key=lambda ms: ms[1])
        return self.base_model(best_model_and_score[0])

    def split_sequences(self, indices):
        training_sequences = [self.sequences[i] for i in indices]
        training_x = [evt for sequence in training_sequences for evt in sequence]
        training_lengths = [len(sequence) for sequence in training_sequences]
        return training_x, training_lengths

    def iterate_sequences(self, indices):
        sequence, lengths = self.split_sequences(indices)
        begin = 0
        for length in lengths:
            yield sequence[begin:begin + length]
            begin = begin + length

    def scoreTrial(self, nr_components):
        nr_cross_validation_sets = min(3, len(self.sequences))
        splits = KFold(n_splits=nr_cross_validation_sets).split(self.sequences)

        # the mean of all cross-validation folds for the scored nr_components
        scores = list(map(lambda split: self.scoreTrialWithFold(split, nr_components), splits))
        flattened_scores = [val for sublist in scores for val in sublist]
        return nr_components, statistics.mean(flattened_scores)

    def scoreTrialWithFold(self, fold_split, nr_components):
        train_indices = fold_split[0]
        test_indices = fold_split[1]

        # training using 'training part' of test-set
        training_x, training_lengths = self.split_sequences(train_indices)
        model = self.create_and_fit_model(nr_components, training_x, training_lengths)
        if not model:
            yield -math.inf
        else:
            # score using 'test part' of test-set
            for test_x in self.iterate_sequences(test_indices):
                try:
                    yield self.score(model, test_x)
                except ValueError:
                    # invalid model
                    yield -math.inf

    @abstractmethod
    def score(self, model, x):
        """
        Generates the score for a given model using the observations x
        :param model: the model being scored
        :param x: observation X to be scored by the model
        :return: calculated score. The model with highest score is the best.
        """
        pass


class SelectorBIC(ModelSelectorUsingCV):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def score(self, model, x):
        """
       Generates the score for a given model using the observations x
       :param model: the model being scored
       :param x: observation X to be scored by the model
       :return: calculated score. The model with highest score is the best.
       """
        log_likelihood = model.score(x, [len(x)])
        nr_components = model.n_components
        nr_observations = len(x)
        negative_bic = 2 * log_likelihood - nr_components * math.log(nr_observations)
        return negative_bic  # a negative bic is returned because max score is considered best


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelectorUsingCV):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def score(self, model, x):
        """
       Generates the score for a given model using the observations x
       :param model: the model being scored
       :param x: observation X to be scored by the model
       :return: calculated score. The model with highest score is the best.
       """
        log_likelihood = model.score(x, [len(x)])
        return log_likelihood
