from model import create_model
from preprocessiong import preprocess_data

X, y = preprocess_data()

X_train, y_train = X[:970], y[:, :970]
X_test, y_test = X[970:], y[:, 970:]

model = create_model()
model.summary()
hist = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, epochs=30, verbose=1, validation_split=0.2)

model.save("captcha_model.h5")
