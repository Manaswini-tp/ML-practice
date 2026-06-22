import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Create sample data
np.random.seed(0)

data = np.random.rand(100)

# First 50 points for training
train_data = data[:50]

# Assign labels to training data
train_labels = []

for value in train_data:
    if value <= 0.5:
        train_labels.append("Class1")
    else:
        train_labels.append("Class2")


# Remaining 50 points for testing
test_data = data[50:]


# KNN function
def knn(test_point, k):

    # Calculate distance from all training points
    distances = []

    for i in range(len(train_data)):
        distance = abs(test_point - train_data[i])
        distances.append((distance, train_labels[i]))

    # Sort according to distance
    distances.sort()


    # Select k nearest neighbors
    nearest_neighbors = []

    for i in range(k):
        nearest_neighbors.append(distances[i][1])


    # Find majority class
    result = Counter(nearest_neighbors).most_common(1)

    return result[0][0]


# Choose value of K
k = 3


# Predict classes
predictions = []

for point in test_data:
    predictions.append(knn(point, k))


# Display predictions
for i in range(len(test_data)):
    print("Test Point:", round(test_data[i],4),
          "Predicted Class:", predictions[i])


# Plot graph

plt.figure(figsize=(8,6))


# Plot training data
for i in range(len(train_data)):

    if train_labels[i] == "Class1":
        color = "blue"
    else:
        color = "red"

    plt.scatter(train_data[i], 0, color=color)


# Plot test predictions
for i in range(len(test_data)):

    if predictions[i] == "Class1":
        color = "blue"
    else:
        color = "red"

    plt.scatter(test_data[i], 1, color=color, marker='x')


plt.title("KNN Classification")
plt.xlabel("Data Points")
plt.ylabel("Class")

plt.grid()
plt.show()
