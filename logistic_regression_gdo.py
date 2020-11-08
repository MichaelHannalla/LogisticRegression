# CSE488 - Computational Intelligence
# Faculty of Engineering - Ain Shams University
# Bipolar Classifer Project - MNIST Dataset
# Haidy Sorial Samy, 16P8104 | Michael Samy Hannalla, 16P8202
# Training of bi-polar perceptron using ordinary gradient descent of our own implementation

import numpy as np
import matplotlib.pyplot as plt

class Bipolar_Perceptron():

    # Initialization of perceptron, initializing its parameter vector using random values
    def __init__(self, name, w_shape):
        self.name = name
        self.w_shape = w_shape
        self.parameter_vector = np.random.rand(w_shape)
        self.costfunc_list = []
        self.errorgrad_norm_list = []
    
    # Defining the perceptron output/activation function
    def perceptronfunc(self,x_input):
        return np.sign(np.dot(self.parameter_vector,x_input))
        
    # Defining the error gradient function
    def errorgradfunc(self, x_inputs, y_labels):
        self.tile_y = np.tile(y_labels,(self.w_shape, 1))
        self.errored = [1 if self.perceptronfunc(x_inputs[i])!= y_labels[i] else 0 for i,_ in enumerate(x_inputs)]
        self.error_grad = -1 * np.multiply(self.tile_y.T, x_inputs) 
        self.error_grad = np.matmul(np.reshape(self.errored,(1, len(self.errored))), self.error_grad)
        return np.reshape(self.error_grad,(self.w_shape,)) / len(x_inputs)
    
    # Defining the error (cost) function
    def errorfunc(self, x_inputs, y_labels):
        self.errored = [1 if self.perceptronfunc(x_inputs[i])!= y_labels[i] else 0 for i,_ in enumerate(x_inputs)]
        self.cost = -y_labels * np.dot(x_inputs,self.parameter_vector)
        return np.sum(np.multiply(self.errored, self.cost)) / len(x_inputs)
    
    # Efficiency function for testing of perceptron on testing dataset
    def efficiencyfunc(self, x_inputs, y_labels):
        self.success = [1 if self.perceptronfunc(x_inputs[i])== y_labels[i] else 0 for i,_ in enumerate(x_inputs)]
        return np.sum(self.success) / len(x_inputs)
    
    # Defining the perceptron training function by gradient descent optimization
    def train(self, x_inputs, y_labels, learning_rate, gradient_tolerance):
        print("Started gradient learning of bipolar perceptron")
        self.errorgrad_norm = np.linalg.norm(self.errorgradfunc(x_inputs, y_labels))
        self.gradient_iteration = 0
        while (self.errorgrad_norm > gradient_tolerance):
            self.errorgrad_norm_list.append(self.errorgrad_norm)
            self.costfunc_list.append(self.errorfunc(x_inputs, y_labels))
            print("Now in gradient iteration: " + str(self.gradient_iteration) + "   Error gradient magnitude = " + str(self.errorgrad_norm))
            self.parameter_vector = self.parameter_vector - self.errorgradfunc(x_inputs, y_labels) * learning_rate
            self.errorgrad_norm = np.linalg.norm(self.errorgradfunc(x_inputs, y_labels))
            self.gradient_iteration += 1
        print("Perceptron training finished")


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
MyPerceptron.train(train_imgs_x, train_labels_y, 7, 0.005)

print("Finished learning and starting testing the model.")
print("Costfunction value on testing dataset = " + str(MyPerceptron.errorfunc(test_imgs_x, test_labels_y)))
print("Perceptron efficiency on testing dataset = " + str(MyPerceptron.efficiencyfunc(test_imgs_x, test_labels_y) * 100) + " %")

plt.plot(range(len(MyPerceptron.costfunc_list)),MyPerceptron.costfunc_list)
plt.xlabel('Gradient Iterations')  
plt.ylabel('Cost Function | Perceptron Error')  
plt.title('Gradient Descent Training of Bipolar Perceptron')
plt.show()
