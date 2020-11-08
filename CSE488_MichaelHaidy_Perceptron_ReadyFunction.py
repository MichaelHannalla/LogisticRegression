# CSE488 - Computational Intelligence
# Faculty of Engineering - Ain Shams University
# Bipolar Classifer Project - MNIST Dataset
# Haidy Sorial Samy, 16P8104 | Michael Samy Hannalla, 16P8202
# Training of bi-polar perceptron using a ready made optimization function

import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Bipolar_Perceptron():
    
    # Initialization of perceptron, initializing its parameter vector using random values
    def __init__(self, name, w_shape):
        self.name = name
        self.w_shape = w_shape
        self.parameter_vector = np.random.rand(w_shape)
        self.costfunc_list = [] # To store the costfunction and error gradient values for each iteration.
        self.errorgrad_norm_list = []
    
    # Defining the perceptron output/activation function
    def perceptronfunc(self, w_vector, *args):
        return np.sign(np.dot(w_vector,args[0]))
    
    # Defining the error gradient function
    def errorgradfunc(self, w_vector, *args):
        self.tile_y = np.tile(args[1],(self.w_shape, 1))
        self.errored = [1 if self.perceptronfunc(self.parameter_vector,args[0][i])!= args[1][i] else 0 for i,_ in enumerate(args[0])]
        self.error_grad = -1 * np.multiply(self.tile_y.T, args[0]) 
        self.error_grad = np.matmul(np.reshape(self.errored,(1, len(self.errored))), self.error_grad)
        return np.reshape(self.error_grad,(self.w_shape,)) / len(args[0])

    # Defining the error (cost) function
    def errorfunc(self, w_vector, *args):
        self.errored = [1 if self.perceptronfunc(w_vector,args[0][i])!= args[1][i] else 0 for i,_ in enumerate(args[0])]
        self.cost = -args[1] * np.dot(args[0],w_vector)
        return np.sum(np.multiply(self.errored, self.cost)) / len(args[0])
    
    # Efficiency function for testing of perceptron on testing dataset
    def efficiencyfunc(self, x_inputs, y_labels):
        self.success = [1 if self.perceptronfunc(self.parameter_vector,x_inputs[i]) == y_labels[i] else 0 for i,_ in enumerate(x_inputs)]
        return np.sum(self.success) / len(x_inputs)
    
# Importing of dataset.
print("Started importing of datasets.")
train_data = np.loadtxt('dataset/mnist_train.csv', dtype = np.float16, delimiter=",")
test_data = np.loadtxt('dataset/mnist_test.csv', dtype = np.float16, delimiter=",")
print("Finished importing of datasets.")

class_one = 0     # In our case we used class one to be the zero digit
class_two = 1     # And class two to be the one digit

# Dataset processing.
training_imgs_labelled = train_data[np.logical_or(train_data[:,0] == class_one, train_data[:,0] == class_two)]
training_imgs_labelled[:,1:] = np.asfarray(training_imgs_labelled[:, 1:]) * 0.99 / 255 + 0.01
training_imgs_labelled = np.where(training_imgs_labelled == class_one, -1, training_imgs_labelled)
training_imgs_labelled = np.where(training_imgs_labelled == class_two,  1, training_imgs_labelled)
train_imgs_x = training_imgs_labelled[:,1:]
train_imgs_x = np.hstack((train_imgs_x, np.ones((len(train_imgs_x),1))))
train_labels_y = training_imgs_labelled[:,0]
del train_data, training_imgs_labelled

testing_imgs_labelled = test_data[np.logical_or(test_data[:,0] == class_one, test_data[:,0] == class_two)]
testing_imgs_labelled[:,1:] = np.asfarray(testing_imgs_labelled[:, 1:]) * 0.99 / 255 + 0.01
testing_imgs_labelled = np.where(testing_imgs_labelled == class_one, -1, testing_imgs_labelled)
testing_imgs_labelled = np.where(testing_imgs_labelled == class_two,  1, testing_imgs_labelled)
test_imgs_x = testing_imgs_labelled[:,1:]
test_imgs_x = np.hstack((test_imgs_x, np.ones((len(test_imgs_x),1))))
test_labels_y = testing_imgs_labelled[:,0]
del test_data, testing_imgs_labelled

MyPerceptron = Bipolar_Perceptron("MNIST Classifier", 785)

# The call back function to append the cost function after each optimization iteration
def opti_callback(xk):
    currenterror = MyPerceptron.errorfunc(xk, *(train_imgs_x, train_labels_y))
    MyPerceptron.costfunc_list.append(currenterror)

print("SLSQP optimization has started.")
result = minimize(MyPerceptron.errorfunc, args= (train_imgs_x, train_labels_y), x0 = MyPerceptron.parameter_vector, method= 'SLSQP', options={'disp':True}, callback= opti_callback, tol = 0.005)
MyPerceptron.parameter_vector = result.x
print("Finished optimization")

print("Finished learning and starting testing the model.")
costfunc_test = MyPerceptron.errorfunc(MyPerceptron.parameter_vector, *(test_imgs_x, test_labels_y))
print("Costfunction value on testing dataset = " + str(costfunc_test))
efficiency_test = MyPerceptron.efficiencyfunc(*(test_imgs_x, test_labels_y)) * 100
print("Perceptron efficiency on testing dataset = " + str(efficiency_test) + " %")

plt.plot(range(len(MyPerceptron.costfunc_list)),MyPerceptron.costfunc_list)
plt.xlabel('Gradient Iterations')  
plt.ylabel('Cost Function | Perceptron Error')  
plt.title('Gradient Descent Training of Bipolar Perceptron')
plt.show()