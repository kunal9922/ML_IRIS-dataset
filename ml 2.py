from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
# load data sets of iris
iris = load_iris()
feature = iris.data # feature of iris
labels = iris.target # labels 0f iris

# split  data into training and testing data
feature_train, feature_test, labels_train, labels_test = train_test_split(feature, labels, test_size = .5)
  
# classifier TREE
clf = tree.DecisionTreeClassifier()
clf.fit(feature_train,labels_train)
pred = clf.predict(feature_test)
print(pred)

# for finding accracy of a program
from sklearn.metrics import accuracy_score
print(accuracy_score(labels_test, pred))

