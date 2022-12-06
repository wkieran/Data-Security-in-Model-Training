from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

import matplotlib.pyplot as plt
#plt.gray()
fig, ax = plt.subplots(2, 2)
#plt.matshow(digits.images[2])

fig.suptitle('Random')
ax[0,0].matshow(digits.images[2])
ax[0,1].matshow(digits.images[0])
ax[1,0].matshow(digits.images[5])
ax[1,1].matshow(digits.images[7])

plt.show()