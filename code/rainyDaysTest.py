import numpy as np
import matplotlib.pyplot as plt


mu, sigma = 5, 1 # mean and standard deviation
data1 = np.random.normal(mu, sigma, 200)
mu2, sigma = 6, 1 # mean and standard deviation
data2 = np.random.normal(mu2, sigma, 200)


plt.figure()

count, bins, ignored = plt.hist(data1, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')

count, bins, ignored = plt.hist(data2, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu2)**2 / (2 * sigma**2) ),
         linewidth=2, color='g')

plt.show()


data3 = np.ones(data1.shape)*100
data3 = np.subtract(data3, data1)

data4 = np.ones(data2.shape)*100
data4 = np.subtract(data4, data2)

plt.figure()

count, bins, ignored = plt.hist(data3, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')

count, bins, ignored = plt.hist(data4, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu2)**2 / (2 * sigma**2) ),
         linewidth=2, color='g')

plt.show()
