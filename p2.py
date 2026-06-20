import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)
data = california_housing.frame
corr_mat = data.corr()

plt.figure(figsize=(10,6))
sns.heatmap(corr_mat, cmap='coolwarm', annot=True, linewidth=0.5)
plt.title('Correlation Matrix for California housing Dataset')
plt.show()

sns.pairplot(data)
plt.title('Pair Plot for California housing Dataset')
plt.show()
