import sklearn


from scipy import stats
from sklearn.svm import SVC # "Support vector classifier"
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn import datasets

#Load dataset
boston = datasets.load_breast_cancer()
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.1,random_state=109) 


#Defining our SVM classifier
model = SVC(kernel='linear', C=1E10)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Finding the Accuracy

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

