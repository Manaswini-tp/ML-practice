import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,2*np.pi,100)
y = np.sin(x)+0.1*np.random.randn(100)

def locally_weighted_regression(xi, x, y, tau=0.5):
  weights = np.exp(-(xi-x)**2/(2*tau**2))
  W = np.diag(weights)
  x_mat = np.c_[np.ones(len(x)),x]
  theta = np.linalg.pinv(x_mat.T @ W @ x_mat) @ (x_mat.T @ W @ y)
  return np.array([1,xi])@theta

x_test = np.linspace(0,2*np.pi,100)
y_pred = np.array([locally_weighted_regression(xt,x,y) for xt in x_test])

plt.scatter(x,y,color='red', label='Training data')
plt.plot(x_test, y_pred, color='blue', label='LWR fit for tau=0.5')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Locally Weighted Regression Graph")
plt.legend()
plt.show()
