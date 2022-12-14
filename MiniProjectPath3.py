#MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
#import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import copy


rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

#Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)

def dataset_searcher(number_list,images,labels):

  images_nparray = []
  labels_nparray = []

  for num in number_list:
    labels_nparray.append(labels[np.where(labels == num)[0][0]])
    images_nparray.append(images[np.where(labels == num)[0][0]])

  images_nparray = np.array(images_nparray)
  labels_nparray = np.array(labels_nparray)

  return images_nparray, labels_nparray

def print_numbers(images,labels,title):

  nplots = len(images)
  fig = plt.figure(figsize=(8,8))
  for j in range(nplots):
    plt.subplot(1,nplots,j+1)
    plt.imshow(images[j], cmap='binary')
    plt.title(labels[j])
  
  fig.suptitle(title)
  plt.show()

  pass

class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher(class_numbers, images, labels)
#Part 2
print_numbers(class_number_images , class_number_labels, "Specific Number Classes")


model_1 = GaussianNB()

#however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

#Now we can fit the model
model_1.fit(X_train_reshaped, y_train)
#Part 3 Calculate model1_results using model_1.predict()
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
model1_results = model_1.predict(X_test_reshaped)

def OverallAccuracy(results, actual_values):
  #Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
  return metrics.accuracy_score(y_true = actual_values, y_pred = results)


# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))

#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)

#labels using NB model
print_numbers(allnumbers_images , model_1.predict(allnumbers_images.reshape(allnumbers_images.shape[0], -1)), "GNB Model")


#Part 6
#Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)
model_2.fit(X_train_reshaped, y_train)
model2_results = model_2.predict(X_test_reshaped)
Model2_Overall_Accuracy = OverallAccuracy(model2_results, y_test)
print("The overall results of the KNN model is " + str(Model2_Overall_Accuracy))

#labels for all numbers using KNN model
print_numbers(allnumbers_images , model_2.predict(allnumbers_images.reshape(allnumbers_images.shape[0], -1)), "KNN Model")

#Repeat for the MLP Classifier
model_3 = MLPClassifier(random_state=0)
model_3.fit(X_train_reshaped, y_train)
model3_results = model_3.predict(X_test_reshaped)
Model3_Overall_Accuracy = OverallAccuracy(model3_results, y_test)
print("The overall results of the MLP model is " + str(Model3_Overall_Accuracy))

#labels for all numbers using MLP model
print_numbers(allnumbers_images , model_3.predict(allnumbers_images.reshape(allnumbers_images.shape[0], -1)), "MLP Model")

#Part 8
#Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison


#Part 9-11
#Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train
# NB model with poisoned data
model_GNB_poison = GaussianNB()
X_poison_reshaped = X_train_poison.reshape(X_train_poison.shape[0], -1)
model_GNB_poison.fit(X_poison_reshaped, y_train)
GNBp_results = model_GNB_poison.predict(X_test_reshaped)
GNBp_accuracy = OverallAccuracy(GNBp_results, y_test)

print("The poisoned results of the Gaussian model is " + str(GNBp_accuracy))
print_numbers(allnumbers_images, model_GNB_poison.predict(allnumbers_images.reshape(allnumbers_images.shape[0], -1)), "Poisoned GNB Model")

# KNN model with poisoned data
model_KNN_poison = KNeighborsClassifier(n_neighbors=10)
model_KNN_poison.fit(X_poison_reshaped, y_train)
KNNp_results = model_KNN_poison.predict(X_test_reshaped)
KNNp_accuracy = OverallAccuracy(KNNp_results, y_test)

print("The poisoned results of the KNN model is " + str(KNNp_accuracy))
print_numbers(allnumbers_images, model_KNN_poison.predict(allnumbers_images.reshape(allnumbers_images.shape[0], -1)), "Poisoned KNN Model")

# MLP model with poisoned data
model_MLP_poison = MLPClassifier(random_state=0)
model_MLP_poison.fit(X_poison_reshaped, y_train)
MLPp_results = model_MLP_poison.predict(X_test_reshaped)
MLPp_accuracy = OverallAccuracy(MLPp_results, y_test)

print("The poisoned results of the MLP model is " + str(MLPp_accuracy))
print_numbers(allnumbers_images, model_MLP_poison.predict(allnumbers_images.reshape(allnumbers_images.shape[0], -1)), "Poisoned MLP Model")

#Part 12-13
# Denoise the poisoned training data, X_train_poison. 
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data. 
# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64

kernel_pca = KernelPCA(n_components=400, kernel="rbf", gamma=1e-3, fit_inverse_transform=True, alpha=5e-3)
kernel_pca.fit(X_poison_reshaped)
X_train_denoised = kernel_pca.inverse_transform(kernel_pca.transform(X_poison_reshaped))


#Part 14-15
#Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
#Explain how the model performances changed after the denoising process.

# NB model with denoised data
model_GNB_denoised = GaussianNB()
X_denoised_reshaped = X_train_denoised.reshape(X_train_denoised.shape[0], -1)
model_GNB_denoised.fit(X_denoised_reshaped, y_train)
GNBd_results = model_GNB_denoised.predict(X_test_reshaped)
GNBd_accuracy = OverallAccuracy(GNBd_results, y_test)

print("The denoised results of the Gaussian model is " + str(GNBd_accuracy))
print_numbers(allnumbers_images, model_GNB_denoised.predict(allnumbers_images.reshape(allnumbers_images.shape[0], -1)), "Denoised GNB Model")

# KNN model with denoised data
model_KNN_denoised = KNeighborsClassifier(n_neighbors=10)
model_KNN_denoised.fit(X_denoised_reshaped, y_train)
KNNd_results = model_KNN_denoised.predict(X_test_reshaped)
KNNd_accuracy = OverallAccuracy(KNNd_results, y_test)

print("The denoised results of the KNN model is " + str(KNNd_accuracy))
print_numbers(allnumbers_images, model_KNN_denoised.predict(allnumbers_images.reshape(allnumbers_images.shape[0], -1)), "Denoised KNN Model")

# MLP model with denoised data
model_MLP_denoised = MLPClassifier(random_state=0)
model_MLP_denoised.fit(X_denoised_reshaped, y_train)
MLPd_results = model_MLP_denoised.predict(X_test_reshaped)
MLPd_accuracy = OverallAccuracy(MLPd_results, y_test)

print("The denoised results of the MLP model is " + str(MLPd_accuracy))
print_numbers(allnumbers_images, model_MLP_denoised.predict(allnumbers_images.reshape(allnumbers_images.shape[0], -1)), "Denoised MLP Model")