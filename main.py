from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import cifar100
from keras.utils import to_categorical
import keras

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)

def reshapeX(x):
    dim = ( x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
    return keras.utils.normalize(x.reshape(dim))

x_train = reshapeX(x_train)
x_test = reshapeX(x_test)
model = Sequential()


for i in range(20):
    model.add(Dense(units=100, activation='relu', input_dim=x_train.shape[1], kernel_initializer='he_normal'))
model.add(Dense(units=100, activation='softmax', kernel_initializer='he_normal'))

#sgd = keras.optimizers.SGD(lr=2.0, momentum=0.0, decay=0.0, nesterov=False)

model.compile(loss='categorical_crossentropy',
              #optimizer=sgd,
              optimizer='sgd',
              metrics=['accuracy'])

print(x_train.shape, y_train.shape)
model.fit(x_train, y_train, epochs=20, batch_size=32)

loss, acc = model.evaluate(x_test, y_test, batch_size=128)

print("loss", loss, "acc", acc)

#classes = model.predict(x_test, batch_size=128)