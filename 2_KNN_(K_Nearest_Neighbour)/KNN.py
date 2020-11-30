import sklearn
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('car.data')
print(data.head())

# we got too many string values and we are dealing with numbers so we have to convert those values into numbers
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=5)     # specifying neighbours
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print(accuracy)

predicted = model.predict(x_test)
names = ["unacc", "", "acc", "good", "vgood"]
for x in range(len(x_test)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])




