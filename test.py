from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

import matplotlib.pyplot as plt
#plt.gray()
plt.matshow(digits.images[2])

plt.show()