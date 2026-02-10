from keras.datasets import mnist 
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping , ModelCheckpoint
from keras.models import Sequential 
from keras.layers import Dense
from keras.models import load_model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test,10)

model = Sequential()

model.add(Dense(512, activation = "relu" , input_shape = (28*28,)))

model.add(Dense(256, activation = "tanh"))

model.add(Dense(10, activation = "softmax"))

model.summary()

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"])

early_stopping = EarlyStopping(monitor = "val_loss",
                               patience = 3,
                               restore_best_weights=True
                               )

checkpoint = ModelCheckpoint("ann_bestmodel.h5", monitor = "val_loss", save_best_only=True)

model.fit(x_train,y_train,
          epochs = 10,
          batch_size=60,
          validation_split = 0.2,
          callbacks=[early_stopping, checkpoint])


model.save("final_mnist_ann_model.h5")
loaded_model = load_model("final_mnist_ann_model.h5")

loss, accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
print("\n--- Test Sonuçları ---")
print(f"Test Kaybı (Loss): {loss:.4f}")
print(f"Test Doğruluğu (Accuracy): {accuracy:.4f}")