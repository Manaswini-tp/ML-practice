import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
x, y = iris.data, iris.target
x_pca = PCA(n_components=2).fit_transform(x)

colors = ['violet', 'blue', 'green']
for i in range(3):
  plt.scatter(x_pca[y==i,0], x_pca[y==i,1], color=colors[i], label=iris.target_names[i])

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")
plt.legend()
plt.grid()
plt.show()
