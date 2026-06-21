import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

cancer_data = load_breast_cancer()
x, y = cancer_data.data, cancer_data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy*100:.2f}%")
new_sample = np.array([x_test[0]])
prediction = clf.predict(new_sample)
if prediction==1:
  class_label="Benign"
else:
  class_label="Malignant"
print("Class label: ",class_label)

plt.figure(figsize=(12,8))
tree.plot_tree(
  clf,
  filled=True,
  feature_names = cancer_data.feature_names,
  class_names = cancer_data.target_names
)
plt.show()
