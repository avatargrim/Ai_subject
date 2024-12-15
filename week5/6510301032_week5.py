import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam

# 1. สร้างข้อมูล 2 กลุ่มด้วย make_blobs
X, y = make_blobs(n_samples=200, centers=[[2, 2], [3, 3]], cluster_std=0.75, random_state=42)

# 2. แบ่งข้อมูลสำหรับ Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 3. สร้างโมเดล Neural Network
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])

# 4. เทรนโมเดล
model.fit(X_train, y_train, epochs=300, batch_size=200, verbose=0)

# 5. สร้าง Decision Boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z=z.reshape(xx.shape)

plt.contourf(xx, yy, z, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.5)
plt.contour(xx, yy, z, levels=[0.5], colors='black', linewidths=2)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('Decision Boundary with Separation Line')
plt.show()