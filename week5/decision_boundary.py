import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam
import pandas as pd

# Load the Titanic dataset
url = "week5/titanic.csv"
data = pd.read_csv(url)

# Selecting features and target variable
X = data[['Age', 'Fare']].values
y = data['Survived'].values

# Handling missing values by replacing them with the min value
X[np.isnan(X)] = np.nanmin(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = Sequential()
model.add(Dense(16, input_dim=2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
optimizer = (Adam(0.0001))
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=300,batch_size=200,verbose=0)


y_pred_prob=model.predict(X_test)
y_pred= np.round(y_pred_prob).astype(int).ravel()

x_min,x_max = X[:,0].min() - 1, X[:,0].max()+1
y_min,y_max = X[:,0].min() - 1, X[:,0].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))

z=model.predict(np.c_[xx.ravel(),yy.ravel()])
z=np.round(z).astype(int)
z=z.reshape(xx.shape)

plt.contourf(xx,yy,z,alpha=0.4)
plt.scatter(X[:,0], X[:,1],c=y,s=20,edgecolors='k')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Decicion Boundary')
plt.show()