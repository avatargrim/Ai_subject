import numpy as np
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

# 3. สร้างโมเดล Neural Network ที่จำลอง Linear Classifier
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))  # ชั้นเดียว Output Linear

# คอมไพล์โมเดล
model.compile(loss='binary_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])

# 4. เทรนโมเดล
model.fit(X_train, y_train, epochs=300, batch_size=10, verbose=0)

# 5. สร้าง Decision Boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# ทำนายผลลัพธ์สำหรับแต่ละจุด
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 6. Plot Decision Boundary
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.5)  # พื้นที่สองสี
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)  # เส้นแบ่งสีดำ
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')  # จุดข้อมูล
plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('Decision Boundary (Neural Network Linear)')
plt.show()
