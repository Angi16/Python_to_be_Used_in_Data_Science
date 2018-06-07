from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Data : [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],[190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

# Labels : [male/female]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female','female', 'male', 'male']

#I will use 4 classifiers here to compare which one produces better result 

tree_clf = tree.DecisionTreeClassifier()
svm_clf = SVC()
perceptron_clf = Perceptron()
KNN_clf = KNeighborsClassifier()

# Now I will use these 4 classifiers to train our data

tree_clf.fit(X, Y)
svm_clf.fit(X, Y)
perceptron_clf.fit(X, Y)
KNN_clf.fit(X, Y)

# Now I will test them using the same data
pred_tree = tree_clf.predict(X)
acc_tree = accuracy_score(Y, pred_tree) * 100
print('Accuracy for DecisionTree: {}'.format(acc_tree))

pred_svm = svm_clf.predict(X)
acc_svm = accuracy_score(Y, pred_svm) * 100
print('Accuracy for SVM: {}'.format(acc_svm))

pred_per = perceptron_clf.predict(X)
acc_per = accuracy_score(Y, pred_per) * 100
print('Accuracy for perceptron: {}'.format(acc_per))

pred_KNN = KNN_clf.predict(X)
acc_KNN = accuracy_score(Y, pred_KNN) * 100
print('Accuracy for KNN: {}'.format(acc_KNN))

# The best classifier is selected from svm, per, KNN,  tree
index = np.argmax([acc_svm, acc_per, acc_KNN])
classifiers = {0: 'SVM', 1: 'Perceptron', 2: 'KNN', 3: 'tree'}
print('Best gender classifier is {}'.format(classifiers[index]))