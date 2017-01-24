import cPickle
import matplotlib.pyplot as plt

saved_data = cPickle.load(open('save_features.pkl', 'r'))
features = saved_data['validation_results']['valid1']['features']
plt.imshow(features[0])
