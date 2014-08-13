"""
Description: Generate a brain activity plot from the data
Author: Triskelion <info@mlwave.com>
Kaggle contest description, rules and data: 
http://www.kaggle.com/c/decoding-the-human-brain/
Code description:
http://mlwave.com/predict-visual-stimuli-from-human-brain-activity/
"""

import config
import matplotlib

#matplotlib.use('qt4agg')
import numpy
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat(config.train_folder + 'train_subject01.mat', squeeze_me=True)
XX = data['X']
# m = XX.mean(0)
# XX -= m
# XX = numpy.nan_to_num(XX / XX.std(0))


fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

XX = XX[:, 2::3, :]
for i in range(XX.shape[1]):
    ax1.plot([f for f in xrange(375)], XX[0, i])

for i in range(XX.shape[1]):
    ax2.plot([f for f in xrange(375)], XX[188, i])

#
# ax2.plot([f for f in xrange(375)], XX[188][cannal])
# ax2.plot([f for f in xrange(375)], XX[188][cannal + 1])
# ax2.plot([f for f in xrange(375)], XX[188][cannal + 2])

plt.show()