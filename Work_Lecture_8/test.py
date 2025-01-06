import numpy as np
import matplotlib.pyplot as plt
import keras.api.models as mod
import keras.api.layers as lay


pitch = 20
step = 1
N = 100
n_train = int(N * 0.7)  # 70% for Training set

def gen_data(x):
    return (x % pitch) / pitch

t = np.arange(1, N + 1)
y = [gen_data(i) for i in t]
y = np.array(y)

# plt.figure()
# plt.plot(y)
# plt.show()

def convertToMatrix(data, step=1):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d])
        Y.append(data[d])
    return np.array(X), np.array(Y)

y = np.sin(0.05 * t * 10) + 0.8 * np.random.rand(N)

train, test = y[:n_train], y[n_train:N]

x_train, y_train = convertToMatrix(train, step)
x_test, y_test = convertToMatrix(test, step)

print("Dimension (Before): ", train.shape, test.shape)
print("Dimension (After): ", x_train.shape, x_test.shape)



model = mod.Sequential()
model.add(lay.SimpleRNN(units=32, input_shape=(step, 1), activation="relu"))
model.add(lay.Dense(units=1))

model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
hist = model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=1)  # เพิ่ม epochs
# คาดการณ์ข้อมูลด้วยโมเดล
y_pred = model.predict(x_test).flatten()

# ผสานข้อมูล Train และ Test สำหรับ Plot เปรียบเทียบ

y_full_pred = np.concatenate([y_train, y_pred])

# Plot การเปรียบเทียบ
plt.figure()
plt.plot( y, label="Original", color="blue")
plt.plot( y_full_pred, 'r--', label="Predict", color="red")
plt.axvline(x=len(train), color="purple", linestyle="-", label="Train-Test Split")
plt.legend()
plt.show()
