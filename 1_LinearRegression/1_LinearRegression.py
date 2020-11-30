import pickle
import sklearn
import numpy as np
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.utils import shuffle


data = pd.read_csv('student/student-mat.csv', sep=";")  # sep=";" because csv is separated by ';'
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

'''
best = 0
for _ in range(30):
    # test_train_splitting
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


    # training a model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    # calculating the accuracy
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy
        # saving a model using pickle
        with open("student.pickle", 'wb') as f:
            pickle.dump(linear, f)
'''

pickle_in = open("student.pickle", 'rb')
linear = pickle.load(pickle_in)

# coefficient
print("CO: ", linear.coef_)
print("Intercept: ", linear.intercept_)

prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print(prediction[x], x_test[x], y_test[x])

p = "G1"
style.use("ggplot")     # look graph half descent
plt.scatter(data[p], data[predict])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
