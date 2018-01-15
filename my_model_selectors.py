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

    def score_safely(self, model, x, x_lengths):
        """
        return a score for the model or -inf if model invalid
        """
        try:
            return model.score(x, x_lengths)
        except ValueError:
            # invalid model
            return -math.inf


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class ModelSelectorUsingTrials(ModelSelector, ABC):
    """
    A selector that iteratively tests models and selects the one with
    best score
    """

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        trials = range(self.min_n_components, self.max_n_components)
        models_and_scores = list(map(lambda nr: (nr, self.scoreTrial(nr)), trials))
        best_model_and_score = max(models_and_scores, key=lambda ms: ms[1])
        return self.base_model(best_model_and_score[0])

    @abstractmethod
    def scoreTrial(self, nr_components):
        """
        calculates a score for a model with the given number of components
        :param nr_components: nr of components for the model to be tested
        :return: the score. A higher score is better, -inf if model is invalid
        """
        pass

class SelectorBIC(ModelSelectorUsingTrials):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def scoreTrial(self, nr_components):
        """
        calculates a score for a model with the given number of components
        :param nr_components: nr of components for the model to be tested
        :return: a score, a higher score is better, -inf if model is invalid
        """

        model = self.create_and_fit_model(nr_components, self.X, self.lengths)

        log_likelihood = self.score_safely(model, self.X, self.lengths)
        assert model.covariance_type == "diag", "following calculation holds just for 'diag'"

        # Following formula is minus 1 because one transition probability can be calculated by
        # subtracting others from total probability which is 1
        nr_transition_params = model.n_components * (model.n_components - 1)

        # Per component (state) there is the output distribution per feature. The formula is minus
        # 1 because one feature output probability can be calculated by subtracting others
        # from total output probability which is 1
        nr_output_params = model.n_components * (model.n_features - 1)

        nr_params = nr_transition_params + nr_output_params

        nr_data_points = len(self.X)

        # calculate bic negative because best score is highest by definition
        negative_bic = 2 * log_likelihood - nr_params * math.log(nr_data_points)
        return negative_bic


class SelectorDIC(ModelSelectorUsingTrials):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def scoreTrial(self, nr_components):
        """
        calculates a score for a model with the given number of components
        :param nr_components: nr of components for the model to be tested
        :return: the score. A higher score is better, -inf if model is invalid
        """
        words = self.words.keys()
        other_words = [k for k in words if k != self.this_word]

        model = self.create_and_fit_model(nr_components, self.X, self.lengths)
        log_likelihoods = self.score_safely(model, self.X, self.lengths)

        def calculate_likelihood_of_word(w):
            x, x_lengths = self.hwords[w]
            return self.score_safely(model,x,x_lengths)

        other_log_likelihood = 1 / (len(words) - 1) * sum(map(calculate_likelihood_of_word, other_words))

        return log_likelihoods - other_log_likelihood


class SelectorCV(ModelSelectorUsingTrials):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''
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
        return statistics.mean(scores)

    def scoreTrialWithFold(self, fold_split, nr_components):
        train_indices = fold_split[0]
        test_indices = fold_split[1]

        # training using 'training part' of test-set
        training_x, training_lengths = self.split_sequences(train_indices)
        model = self.create_and_fit_model(nr_components, training_x, training_lengths)
        if not model:
            return -math.inf
        else:
            # score using 'test part' of test-set
            test_x, test_lengths = self.split_sequences(test_indices)
            return self.score_safely(model, test_x, test_lengths)


