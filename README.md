# Dense Neural Network on Binary Classification with Hyperparameter Tuning
Galen Wilkerson

Problem:  
Supervised classification

Data:
  
The data has 147072 samples with 10 numerical input variables and 1 binary output class.
We note that the input data distribution varies (mean, std) between the columns, therefore it is importance to center and normalize the data.

The output classes are also unbalanced, which can yield poor results.  We will therefore keep track of performance carefully.

We see that only the latter 47072 rows of the data are all NaN values, so we can safely remove them.

We also perform the same checks and scaling on the test (evaluation) data. 

We then split the data randomly into training and validation sets, setting aside 20% of the data for validation.  This allows us to prevent overfitting later.

Methodology:   

Tools:  python 3.6 using jupyter, keras, sklearn, pandas, numpy

Network:
I have decided to use a deep neural network with the keras library.  For this type of problem, either a dense deep learning network or an ensemble method such as Random Forests generally can work well.

The output activation function is sigmoid because we want the output to represent probability distributions and our output is binary (two possible classes).
The binary crossentropy loss function measures the distance between probability distributions, so here I use it to measure the distance between the actual and predicted distribution.
Rmsprop is considered to usually be a good optimizer.

Hyperparameter tuning:

There in addition to the above loss, activation, and optimization methods, there are several important hyperparameters to determine when using deep neural networks.  These are:  Network depth – the number of hidden layers, network capacity – the dimensionality or ‘width’ of hidden layers, and finally any regularization methods to reduce overfitting.

Here, I have explored by hand the network depth and network capacity, which surprisingly revealed that a single-hidden-layer network gave better accuracy than a deeper network.  Also, a capacity of 64 in the hidden layer gave best results.

Finally, despite testing various regularization techniques (L1, L2 and dropout) on both the deeper and 1-hidden-layer network, the best result seemed to be the simple 1-layer network having 64 nodes in the hidden layer, giving a validation accuracy of 0.9378 and a validation loss of 0.1905 at training 160 epochs.  (search for “~BEST MODEL~” in the provided jupyter notebook)

Thoughts:

Given the class imbalance and more time, we could have used precision and recall on the validation data rather than accuracy and loss.  This may help to achieve a better result.ter-Tuning
