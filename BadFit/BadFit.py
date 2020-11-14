import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn import mixture
from matplotlib.colors import LogNorm

f = open('data', 'w')
for i in range(0,1000):
	if (random.random() < 0.5):
		rv = [-np.random.gamma(1.2 ,scale=5),np.random.gamma(1.5 ,scale=10)]
		string =' '.join(["%.9f" % number for number in rv])
		f.write(string+'\n')
	else:
		rv = np.random.multivariate_normal([40, 40], [[10, 4], [4,8]])
		string =' '.join(["%.9f" % number for number in rv])
		f.write(string+'\n')
f.close()

#plt.figure()
with open('data') as f:
	lines = f.readlines()
	x = [line.split()[0] for line in lines]
	y = [line.split()[1] for line in lines]

a=[float(i) for i in x]
b=[float(i) for i in y]

#plt.scatter(a, b, alpha=0.1)
#plt.show()

plt.figure()

data = list(zip(a,b))
clf = mixture.GaussianMixture(n_components=15, covariance_type='full')
data = np.vstack(data)
clf.fit(data)

x = np.linspace(-40., 80.)
y = np.linspace(-20., 100.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(data[:, 0], data[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()