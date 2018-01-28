from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import cifar100
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)

"""
[
    [1],
    [2],
    [3]
]

[
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]
"""

def reshapeX(x):
    dim = ( x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
    return x.reshape(dim)

x_train = reshapeX(x_train)
x_test = reshapeX(x_test)
model = Sequential()


model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(units=100, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print(x_train.shape, y_train.shape)
model.fit(x_train, y_train, epochs=5, batch_size=32)

loss, acc = model.evaluate(x_test, y_test, batch_size=128)

print("loss", loss, "acc", acc)

#classes = model.predict(x_test, batch_size=128)