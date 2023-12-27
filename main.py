import pandas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# load dataset and split into training and testing data
# Dataset Info: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

normdata = StandardScaler()
x_train = normdata.fit_transform(x_train)
x_test = normdata.transform(x_test)

model = Sequential()
input_layer = Dense(10, input_dim=13, activation="relu")
hidden_layer = Dense(15, activation="relu")
output_layer = Dense(1)

model.add(input_layer)
model.add(hidden_layer)
model.add(output_layer)
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

history = model.fit(x_train, y_train, epochs=100)

plt.plot(history.history["mae"])

model.evaluate(x_test, y_test)

predictions = model.predict(x_test)
plt.scatter(y_test, predictions)
plt.xlabel("Expected Values")
plt.ylabel("Output Values")
limits = [0,50]
plt.plot(limits, limits)










