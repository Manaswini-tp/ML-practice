import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = load_breast_cancer()
x, y = data.data, data.target
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
kmeans = KMeans(n_clusters=2)
y_kmeans = kmeans.fit_predict(x_scaled)

print("Classification Report")
print(classification_report(y, y_kmeans, zero_division=1))
print("Confusion Matrix")
print(confusion_matrix(y, y_kmeans))

x_pca = PCA(n_components=2).fit_transform(x_scaled)
df = pd.DataFrame(x_pca, columns=['PC1', 'PC2'])
df['Clusters'] = y_kmeans
df['True Value'] = y

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Clusters', palette='Set2')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K Means Clustering on Breast Cancer Dataset")
plt.legend(title="Clusters")
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='True Value', palette='coolwarm')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K Means Clustering on Breast Cancer Dataset")
plt.legend(title="Clusters")
plt.show()
