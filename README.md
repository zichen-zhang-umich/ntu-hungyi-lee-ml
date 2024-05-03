The course website: [MACHINE LEARNING 2022 SPRING](https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php)

# 2/18 Lecture 1: Introduction of Deep Learning

## Preparation 1: 預測本頻道觀看人數 (上)-機器學習基本概念簡介

ML is about finding a function.

Besides **regression** and **classification** tasks, ML also focuses on **<u>Structured Learning</u>** -- create something with structure (image, document).

Three main ML tasks:

- Regression
- Classification
- Structured Learning

### Function with unknown parameters

*Guess* a function (**model**) $y=b+wx_1$ (in this case, a ***linear model***) based on **domain knowledge**. $w$ and $b$ are unknown parameters. 

### Define Loss From Training Data

Loss is a function itself of parameters, written as $L(b,w)$​. 

An example loss function is:

$$
L = \frac{1}{N} \sum_{n} e_n
$$

where $e_n$ can be calculated in many ways:

**Mean Absolute Error (MAE)**

$$
e_n = |y_n - \hat{y}_n|
$$

**Mean Square Error (MSE)**

$$
e_n = (y_n - \hat{y}_n)^2
$$

If $y$ and $\hat{y}_n$ are both probability distributions, we can use **cross entropy**.

<img src="assets/image-20240503102744604.png" alt="image-20240503102744604" style="zoom:25%;" />

### Optimization

The goal is to find the best parameter:

$$
w^{*}, b^{*} = \arg\min_{w,b} L
$$

Solution: **Gradient Descent**

1. Randomly pick an intitial value $w^0$
2. Compute $\frac{\partial L}{\partial w} \bigg|_{w=w^0}$ (if it's negative, we should increase $w$; if it's positive, we should decrease $w$​)
3. Perform update step: $w^1 \leftarrow w^0 - \eta \frac{\partial L}{\partial w} \bigg|_{w=w^0}$ iteratively
4. Stop either when we've performed a maximum number of update steps or the update stops ($\eta \frac{\partial L}{\partial w} = 0$)

If we have two parameters:

1. (Randomly) Pick initial values $w^0, b^0$

2. Compute:

   $w^1 \leftarrow w^0 - \eta \frac{\partial L}{\partial w} \bigg|_{w=w^0, b=b^0}$

   $b^1 \leftarrow b^0 - \eta \frac{\partial L}{\partial b} \bigg|_{w=w^0, b=b^0}$​

<img src="assets/image-20240503111251767.png" alt="image-20240503111251767" style="zoom:25%;" />

## Preparation 2: 預測本頻道觀看人數 (下)-機器學習基本概念簡介

### Beyond Linear Curves: Neural Networks

However, linear models are too simple. Linear models have severe limitations called **model bias**.

<img src="assets/image-20240503112224217.png" alt="image-20240503112224217" style="zoom:25%;" />

All piecewise linear curves:

<img src="assets/image-20240503123546349.png" alt="image-20240503123546349" style="zoom:25%;" />

More pieces require more blue curves.

<img src="assets/image-20240503123753582.png" alt="image-20240503123753582" style="zoom:25%;" />

How to represent a blue curve (**Hard Sigmoid** function): **Sigmoid** function

$$
y = c\:\frac{1}{1 + e^{-(b + wx_1)}} = c\:\text{sigmoid}(b + wx_1)
$$

We can change $w$, $c$ and $b$ to get sigmoid curves with different shapes.

Different Sigmoid curves -> Combine to approximate different piecewise linear functions -> Approximate different continuous functions

<img src="assets/image-20240503124800696.png" alt="image-20240503124800696" style="zoom: 25%;" />

<img src="assets/image-20240503125224141.png" alt="image-20240503125224141" style="zoom: 25%;" />

From a model with high bias $y=b+wx_1$ to the new model with more features and a much lower bias:

$$
y = b + \sum_{i} \; c_i \; \text{sigmoid}(b_i + w_i x_1)
$$

Also, if we consider multiple features $y = b + \sum_{j} w_j x_j$​, the new model can be expanded to look like this:

$$
y = b + \sum_{i} c_i \; \text{sigmoid}(b_i + \sum_{j} w_{ij} x_j)
$$

Here, $i$ represents each sigmoid function and $j$ represents each feature. $w_{ij}$ represents the weight for $x_j$ that corresponds to the $j$-th feature in the $i$-th sigmoid.

<img src="assets/image-20240503131652987.png" alt="image-20240503131652987" style="zoom:33%;" />

<img src="assets/image-20240503132233638.png" alt="image-20240503132233638" style="zoom: 25%;" />

<img src="assets/image-20240503132324232.png" alt="image-20240503132324232" style="zoom: 25%;" />

<img src="assets/image-20240503132402405.png" alt="image-20240503132402405" style="zoom:25%;" />
$$
y = b + \boldsymbol{c}^T \sigma(\boldsymbol{b} + W \boldsymbol{x})
$$

$\boldsymbol{\theta}=[\theta_1, \theta_2, \theta_3, ...]^T$ is our parameter vector:

<img src="assets/image-20240503132744112.png" alt="image-20240503132744112" style="zoom: 25%;" />

Our loss function is now expressed as $L(\boldsymbol{\theta})$.

<img src="assets/image-20240503134032953.png" alt="image-20240503134032953" style="zoom:25%;" />

Optimization is still the same. 

$$
\boldsymbol{\theta}^* = \arg \min_{\boldsymbol{\theta}} L
$$

1. (Randomly) pick initial values $\boldsymbol{\theta}^0$​
2. calculate the gradient $\bold{g} = \begin{bmatrix} \frac{\partial{L}}{\partial{\theta_1}}\bigg|_{\boldsymbol{\theta}=\boldsymbol{\theta}^0} \\ \frac{\partial{L}}{\partial{\theta_2}}\bigg|_{\boldsymbol{\theta}=\boldsymbol{\theta}^0} \\ \vdots \end{bmatrix} = \nabla L(\boldsymbol{\theta}^0)$ with $\boldsymbol{\theta}, \boldsymbol{g} \in \mathbb{R}^n$
3. perform the update step: $\begin{bmatrix} \theta_1^1 \\ \theta_2^1 \\ \vdots \end{bmatrix} \leftarrow \begin{bmatrix} \theta_1^0 \\ \theta_2^0 \\ \vdots \end{bmatrix} - \begin{bmatrix} \eta \frac{\partial{L}}{\partial{\theta_1}}\bigg|_{\boldsymbol{\theta}=\boldsymbol{\theta}^0} \\ \eta \frac{\partial{L}}{\partial{\theta_2}}\bigg|_{\boldsymbol{\theta}=\boldsymbol{\theta}^0} \\ \vdots \end{bmatrix}$, namely $\boldsymbol{\theta}^1 \leftarrow \boldsymbol{\theta}^0 - \eta \boldsymbol{g}$

The terms ***batch*** and ***epoch*** are different.

<img src="assets/image-20240503155656244.png" alt="image-20240503155656244" style="zoom:33%;" />

$$
\text{num\_updates} = \frac{\text{num\_examples}}{\text{batch\_size}}
$$

Batch size $B$​ is also a hyperparameter. One epoch does not tell the number of updates the training process actually has.

### More activation functions: RELU

<img src="assets/image-20240503160800010.png" alt="image-20240503160800010" style="zoom:25%;" />

It looks kind of like the Hard Sigmoid function we saw earlier:

<img src="assets/image-20240503161144168.png" alt="image-20240503161144168" style="zoom:25%;" />

As a result, we can use these two RELU curves to simulate the same result as Sigmoid curve.

$$
y = b + \sum_{2i} c_i \max(0, b_i + \sum_j w_{ij}x_j)
$$

<img src="assets/image-20240503162214759.png" alt="image-20240503162214759" style="zoom:25%;" />

*But, why we want "Deep" network, not "Fat" network*? Answer to that will be revealed in a later lecture.

# 2/25 Lecture 2: What to do if my network fails to train

## Preparation 1: 機器學習任務攻略

### Frameworks of ML

Training data is $\{(\boldsymbol{x}^1, \hat{y}^1), (\boldsymbol{x}^2, \hat{y}^2), ...,(\boldsymbol{x}^N, \hat{y}^N)\}$. Testing data is $\{ \boldsymbol{x}^{N+1}, \boldsymbol{x}^{N+2}, ..., \boldsymbol{x}^{N+M} \}$​

Traing steps:

1. write a function with unknown parameters, namely $y = f_{\boldsymbol{\theta}}(\boldsymbol{x})$​
2. define loss from training data: $L(\boldsymbol{\theta})$
3. optimization: $\boldsymbol{\theta}^* = \arg \min_{\boldsymbol{\theta}} L$​
4. Use $y = f_{\boldsymbol{\theta}^*}(\boldsymbol{x})$ to label the testing data

### Potential issues during training

Model bias: the potential function set of our model does not even include the desired optimal function/model.

<img src="assets/image-20240503173246516.png" alt="image-20240503173246516" style="zoom: 25%;" />

Large loss doesn't always imply issues with model bias. There may be issues with *optimization*. That is, gradient descent does not always produce global minima. We may stuck at a local minima. In the language of function set, the set theoretically contain optimal function $f^*(\boldsymbol{x})$. However, we may never reach that.

<img src="assets/image-20240503174333051.png" alt="image-20240503174333051" style="zoom: 33%;" />

