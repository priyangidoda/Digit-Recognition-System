# Single Digit Classifier

Artificial Intelligence is a growing field and is in demand across industries. It is a hot topic to read. 
Today's largest and most successful enterprises have used AI to improve their operations and gain advantage on their competitors.
This is my first CS project for  introduction to Deep Learning.

### Abstract

In this project the single digit MNIST dataset to train a classifier to predict which digit is shown in the image using numpy.

To get a deep insight and to understand all the concepts required to build a simple multiclass classifier, the following methods were used to solve the problem.

![](https://i.imgur.com/JyWCpiM.png)


### Key learnings

#### One hot encoding

This is used in categorical data to calculate loss. If labels of categorical data are used to train that may lead to wrong result. 

This is extremely important so that the error between two predictions doesn't depend on the class name but on the probablity score of that class prediction.

Ex. If 8 is  Predicted as 9 or as 3 we have no means to say which is better, because we don't have the measure of how wrong the model is. It is possible that the probablity scores of 8 in the two of these examples are 0.1 and 0.3

#### Numpy functions
A lot of numpy funcitons were used which included
- `reshape()`
- `asarray()`
- `random.randn`

#### Activation functions

Activation funcitons are functions that add non linearity to the Deep Networks. These NON-Linear activation Functions are extremely important as if they aren't used all the network will work as if it had just one linear layer.
For this project Relu activation is used.
Other than relu there are various types of activation functions like tanh, sigmoid, leaky relu etc.

#### Cross Entropy Losses


Also called logarithmic loss, log loss or logistic loss. Each predicted class probability is compared to the actual class desired output 0 or 1 and a score/loss is calculated that penalizes the probability based on how far it is from the actual expected value. The penalty is logarithmic in nature yielding a large score for large differences close to 1 and small score for small differences tending to 0.
Cross-entropy loss is used when adjusting model weights during training. The aim is to minimize the loss, i.e, the smaller the loss the better the model. A perfect model has a cross-entropy loss of 0.
Mathematically it is defined as

![](https://i.imgur.com/fQUKXJH.png)
Numpy code is  `J=-np.sum(np.multiply(Y1,(np.log(Y_hat))))/m`

[Source1](https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e) 
[Source2](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)

### Implementation Details

For training of the network the following steps were taken

#### Initialization
Weights are randomly initialized using numpy.random.randn() funtion of which is then multiplied by a small number to prevent the problem of exploding gradients.

![](https://i.imgur.com/wzj2vBp.png)

[Image source](https://medium.com/@keonyonglee/bread-and-butter-from-deep-learning-by-andrew-ng-course-1-neural-networks-and-deep-learning-41563b8fc5d8)
The biases on the other hand are safely initialised by zeros.

#### Forward Prop

- The input goes into the first layer this output is then passed into the activation layer.
- Then after passing through all the layers, the final result is passed to the softmax layer.
- And then the output is predicted in the form of class probablities.

#### Back Prop

![](https://i.imgur.com/wAQfVdy.png)

[Img source](https://towardsdatascience.com/back-propagation-simplified-218430e21ad0)
 Back-propagation is all about feeding this loss backwards in such a way that we can fine-tune the weights based on which. The optimization function (Gradient Descent in our example) will help us find the weights that will — hopefully — yield a smaller loss in the next iteration.

Diffrential of all the components are computed for backprop. Deltas are calculated saperately and then these deltas are used to tweek the weights and biases in a direction that the model starts performing better.

#### Updating parameters

The weights and biases are updated using the deltas caluclated. Learning rate is defined so that the weights don't shoot up.
Learning rate is usually between 0.1 to 0.01



### Results

This was a learning project so I followed Andrew NG's DeepLearning Specialization. The following loss plot was obtained 

![](https://i.imgur.com/eOjE2K5.png)