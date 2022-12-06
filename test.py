from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

import matplotlib.pyplot as plt
#plt.gray()
fig, ax = plt.subplots(2, 2)
#plt.matshow(digits.images[2])

fig.suptitle('Random')
ax[0,0].matshow(digits.images[8])
ax[0,1].matshow(digits.images[9])
ax[1,0].matshow(digits.images[10])
ax[1,1].matshow(digits.images[11])

plt.show()