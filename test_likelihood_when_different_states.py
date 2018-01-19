from my_recognizer import recognize
from asl_utils import show_errors
from my_model_selectors import *
import numpy as np
import pandas as pd
from asl_data import AslDb
from matplotlib import (cm, pyplot as plt, mlab)

asl = AslDb()  # initializes the database

### Ground Features
features_ground = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

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

training = asl.build_training(features_ground)  # Experiment here with different feature sets defined in part 1
sequences = training.get_all_sequences()
Xlengths = training.get_all_Xlengths()
x, lengths = Xlengths["JOHN"]
warnings.filterwarnings("ignore", category=DeprecationWarning)

for component in range(1,14):

    model = GaussianHMM(n_components=component, covariance_type="diag", n_iter=1000,
                                    random_state=14, verbose=False).fit(x, lengths)
    score = model.score(x,[lengths[0]])
    print(f"Nr:{model.n_components} / Score:{score}")



