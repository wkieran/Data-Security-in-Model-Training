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
  #insert code that when given a list of integers, will find the labels and images
  #and put them all in numpy arrary ****(same time as training and testing data)****
  '''
  images_nparray = np.empty([0,0])
  labels_nparray = np.empty([0,0])

  for n in number_list:
    np.append(images_nparray, images[n])
    np.append(labels_nparray, labels[n])
  '''
  # need some evaluation that goes through number_list. looking for x, 
  # cycle through labels and the image associated with label will be appened
  #np where == ?
  
  images_nparray = np.array(images[number_list])
  labels_nparray = np.array(labels[number_list])

  return images_nparray, labels_nparray

def print_numbers(images,labels):
  #insert code that when given images and labels (of numpy arrays)
  #the code will plot the images and their labels in the title.
  '''
  fig, ax = plt.subplots(1,5)
  for i in range(len(images)):
    #print(images[i])
    ax[0,i].imshow(images[i], cmap='binary')
    ax[0,i].set_title(labels[i])
  plt.show()
  '''

  nplots = len(images)
  fig = plt.figure(figsize=(8,8))
  for j in range(nplots):
      plt.subplot(1,nplots,j+1)
      plt.imshow(images[j], cmap='binary')
      plt.title(labels[j])
  plt.show()


  pass

class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher(class_numbers, images, labels)
#Part 2
print_numbers(class_number_images , class_number_labels)


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
#What should go in here? Hint, look at documentation and some reshaping may need to be done)


def OverallAccuracy(results, actual_values):
  #Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
  return metrics.accuracy_score(y_true = actual_values, y_pred = results)


# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))

#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
#allnumbers = np.array(allnumbers)
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)
#allnumbers_result = model_1.predict(allnumbers.reshape(-1, 1))
#allnumbers_Accuracy = OverallAccuracy(allnumbers_result, y_test)
#print("The overall results of the Gaussian model is " + str(allnumbers_Accuracy))
print_numbers(allnumbers_images , model1_results) #this is with NB model


#Part 6
#Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)
model_2.fit(X_train_reshaped, y_train)
model2_results = model_2.predict(X_test_reshaped)
Model2_Overall_Accuracy = OverallAccuracy(model2_results, y_test)
print("The overall results of the KNN model is " + str(Model2_Overall_Accuracy))

#Repeat for the MLP Classifier
model_3 = MLPClassifier(random_state=0)
model_3.fit(X_train_reshaped, y_train)
model3_results = model_3.predict(X_test_reshaped)
Model3_Overall_Accuracy = OverallAccuracy(model3_results, y_test)
print("The overall results of the MLP model is " + str(Model3_Overall_Accuracy))


#Part 8
#Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison


#Part 9-11
#Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train
model_GNB_poison = GaussianNB()
X_poison_reshaped = X_train_poison.reshape(X_train_poison.shape[0], -1)
model_GNB_poison.fit(X_train_reshaped, y_train)
model1_results = model_1.predict(X_test_reshaped)

print("The poisoned results of the Gaussian model is")


#Part 12-13
# Denoise the poisoned training data, X_train_poison. 
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data. 
# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64

kernel_pca = KernelPCA(n_components=400, kernel="rbf", gamma=1e-3, fit_inverse_transform=True, alpha=5e-3)
kernel_pca.fit(X_poison_reshaped)
X_train_denoised = kernel_pca.inverse_transform(kernel_pca.transform(X_poison_reshaped))
#X_train_denoised = 0# fill in the code here


#Part 14-15
#Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
#Explain how the model performances changed after the denoising process.

