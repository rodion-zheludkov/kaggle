import config
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat(config.train_folder + 'train_subject01.mat', squeeze_me=True)
XX = data['X']
