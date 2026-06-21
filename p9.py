import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB

data = fetch_olivetti_faces()
x, y = data.data, data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}")

print("Classification Report: ")
print(classification_report(y_test, y_pred, zero_division=1))

cross_val = cross_val_score(gnb, x, y)
print(f"Cross Value Score: {cross_val.mean()*100:.2f}%")

plt.figure(figsize=(12,8))
for i in range(15):
  plt.subplot(3,5,i+1)
  plt.imshow(x_test[i].reshape(64,64), cmap='grey')
  plt.title(f"Train: {y_test[i]} Pred: {y_pred[i]}")
  plt.axis('off')
plt.show()
