from sklearn.externals import joblib
from sklearn import datasets
from sklearn.datasets import fetch_mldata
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np

#download the Mnist dataset
mnist = fetch_mldata('MNIST original', transpose_data=True, data_home='scikit_learn_data')

#save the images and labels
features = np.array(mnist.data, 'int16') 
labels = np.array(mnist.target, 'int')

# HOG features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
    list_hog_fd.append(fd)
    #we have to mention block_norm=='L2-Hys' next time
hog_features = np.array(list_hog_fd, 'float64')

#Create a multiclass SVM (Multiclass as there are 10 digits)
clf = LinearSVC()

#Fit the data and compress a bit 3 is a good compromise
clf.fit(hog_features, labels)
joblib.dump(clf, "digits_cls.pkl", compress=3)

