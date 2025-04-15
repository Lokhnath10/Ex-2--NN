<H3>NAME: Lokhnath J</H3>
<H3>REG NO: 212223240079</H3>
<H3>EX NO. 2 </H3>
<H3>DATE:15-04-2025</H3>

## <H1 ALIGN =CENTER>Implementation of Perceptron for Binary Classification</H1>

# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs<BR>
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

# RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.<BR>

A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.<BR>

<b>The important key points to be focused to implement a perceptron:</b><BR>

Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.<BR>

Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.<BR>

Perceptron requires linearly separable samples to achieve convergence.<BR>

<b>The math of Perceptron:</b><BR>

If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’<BR>

f(x)=w.x+b<br>

A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:<br>

<img width="283" alt="image" src="https://github.com/Lavanyajoyce/Ex-2--NN/assets/112920679/c6d2bd42-3ec1-42c1-8662-899fa450f483">

Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.<BR>

# ALGORITHM:

<b>STEP 1:</b> Importing the libraries<BR>
<b>STEP 2:</b> Importing the dataset<BR>
<b>STEP 3:</b> Plot the data to verify the linear separable dataset and consider only two classes<BR>
<b>STEP 4:</b> Convert the data set to scale the data to uniform range by using Feature scaling<BR>
<b>STEP 5:</b> Split the dataset for training and testing<BR>
<b>STEP 6:</b> Define the input vector ‘X’ from the training dataset<BR>
<b>STEP 7:</b> Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
<b>STEP 8:</b> Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’<br>
<b>STEP 9:</b> Assign the learning rate<BR>
<b>STEP 10:</b> For ‘N ‘ iterations ,do the following:<BR>

        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
<b>STEP 11:</b> Plot the error for each iteration <BR>
<b>STEP 12:</b> Print the accuracy<BR>

# PROGRAM:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self._b = 0.0
        self._w = None
        self.misclassified_samples = []

    def fit(self, x: np.array, y: np.array, n_iter=10):
        self._b = 0.0
        self._w = np.zeros(x.shape[1])
        self.misclassified_samples = []
        for _ in range(n_iter):
            errors = 0
            for xi, yi in zip(x, y):
                update = self.learning_rate * (yi - self.predict(xi))
                self._b += update
                self._w += update * xi
                errors += int(update != 0.0)
            self.misclassified_samples.append(errors)

    def f(self, x: np.array) -> float:
        return np.dot(x, self._w) + self._b

    def predict(self, x: np.array):
        return np.where(self.f(x) >= 0, 1, -1)

# Load and prepare the Iris dataset
df = pd.read_csv("iris.csv")
print(df.head())

y = df.iloc[:, 4].values
x = df.iloc[:, 0:3].values

# 3D visualization of the Iris dataset
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Iris data set')
ax.set_xlabel('Sepal length in width (cm)')
ax.set_ylabel('Sepal width in width (cm)')
ax.set_zlabel('Petal length in width (cm)')
ax.scatter(x[:50, 0], x[:50, 1], x[:50, 2], color='red', marker='o', s=4, edgecolor='red', label='Iris Setosa')
ax.scatter(x[50:100, 0], x[50:100, 2], color='blue', marker='^', s=4, edgecolor='blue', label='Iris Versicolour')
ax.scatter(x[100:150, 0], x[100:150, 1], x[100:150, 2], color='green', marker='x', s=4, edgecolor='green', label='Iris Virginica')
plt.legend(loc='upper left')
plt.show()

# Prepare data for binary classification (Setosa vs Versicolour)
x = x[0:100, 0:2]
y = y[0:100]

# 2D visualization of the selected classes
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='Versicolour')
plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.legend(loc='upper left')
plt.show()

# Convert labels to binary values (1 for Setosa, -1 for Versicolour)
y = np.where(y == 'Iris-Setosa', 1, -1)

# Normalize features
x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Train the perceptron
classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train)

# Evaluate the model
print("accuracy", accuracy_score(classifier.predict(x_test), y_test) * 100)

# Plot training errors over epochs
plt.plot(range(1, len(classifier.misclassified_samples) + 1), classifier.misclassified_samples, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.show()
```    

# OUTPUT:

![image](https://github.com/user-attachments/assets/6450f380-eab8-4c2c-9a4c-3c690a3bcb80)

![image](https://github.com/user-attachments/assets/6cbb88fd-6830-467c-805e-effbd1f8c038)

![image](https://github.com/user-attachments/assets/f34c6bb2-7a1a-4b4c-a235-33ff2613e329)

![image](https://github.com/user-attachments/assets/d981f892-1b22-452d-8886-f5cab9785805)

# RESULT:

Thus, a single layer perceptron model is implemented using python to classify Iris data set.
