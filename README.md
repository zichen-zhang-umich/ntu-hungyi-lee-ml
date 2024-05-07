The course website: [MACHINE LEARNING 2022 SPRING](https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php)

# 2/18 Lecture 1: Introduction of Deep Learning

## Preparation 1: ML Basic Concepts

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

## Preparation 2: ML Basic Concepts

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

It looks kind of like the Hard Sigmoid function we saw earlier:

<img src="assets/image-20240503161144168.png" alt="image-20240503161144168" style="zoom:25%;" />

As a result, we can use these two RELU curves to simulate the same result as Sigmoid curve.

$$
y = b + \sum_{2i} c_i \max(0, b_i + \sum_j w_{ij}x_j)
$$

<img src="assets/image-20240503162214759.png" alt="image-20240503162214759" style="zoom:25%;" />

*But, why we want "Deep" network, not "Fat" network*? Answer to that will be revealed in a later lecture.

# 2/25 Lecture 2: What to do if my network fails to train

## Preparation 1: ML Tasks

### Frameworks of ML

Training data is $\{(\boldsymbol{x}^1, \hat{y}^1), (\boldsymbol{x}^2, \hat{y}^2), ...,(\boldsymbol{x}^N, \hat{y}^N)\}$. Testing data is $\{ \boldsymbol{x}^{N+1}, \boldsymbol{x}^{N+2}, ..., \boldsymbol{x}^{N+M} \}$​

Traing steps:

1. write a function with unknown parameters, namely $y = f_{\boldsymbol{\theta}}(\boldsymbol{x})$​
2. define loss from training data: $L(\boldsymbol{\theta})$
3. optimization: $\boldsymbol{\theta}^* = \arg \min_{\boldsymbol{\theta}} L$​
4. Use $y = f_{\boldsymbol{\theta}^*}(\boldsymbol{x})$​ to label the testing data

### General Guide

<img src="assets/image-20240504115031073.png" alt="image-20240504115031073" style="zoom: 25%;" />

### Potential issues during training

Model bias: the potential function set of our model does not even include the desired optimal function/model.

<img src="assets/image-20240503173246516.png" alt="image-20240503173246516" style="zoom: 25%;" />

Large loss doesn't always imply issues with model bias. There may be issues with *optimization*. That is, gradient descent does not always produce global minima. We may stuck at a local minima. In the language of function set, the set theoretically contain optimal function $f^*(\boldsymbol{x})$. However, we may never reach that.

<img src="assets/image-20240503174333051.png" alt="image-20240503174333051" style="zoom: 33%;" />

### Optimization issue

- gain insights from comparison to identify whether the failure of the model is due to optimization issues, overfitting or model bias
- start from shallower networks (or other models like SVM which is much easier to optimize)
- if deeper networks do not contain smaller loss on *training* data,  then there is optimization issues (as seen from the graph below)

<img src="assets/image-20240504100720536.png" alt="image-20240504100720536" style="zoom:25%;" />

For example, here, the 5-layer should always do better or the same as the 4-layer network. This is clearly due to optimization problems.

<img src="assets/image-20240504100948916.png" alt="image-20240504100948916" style="zoom:25%;" />

### Overfitting

Solutions:

- more training data
- data augumentation (in image recognition, it means flipping images or zooming in images)
- use a more constrained model: 
  - less parameters
  - sharing parameters
  - less features
  - early stopping
  - regularization
  - dropout

For example, CNN is a more constrained version of the fully-connected vanilla neural network.

<img src="assets/image-20240504102902784.png" alt="image-20240504102902784" style="zoom:25%;" />

### Bias-Complexity Trade-off

<img src="assets/image-20240504104404308.png" alt="image-20240504104404308" style="zoom:25%;" />

<img src="assets/image-20240504104623658.png" alt="image-20240504104623658" style="zoom:25%;" />

### N-fold Cross Validation

<img src="assets/image-20240504114823462.png" alt="image-20240504114823462" style="zoom:25%;" />

### Mismatch

Mismatch occurs when the training dataset and the testing dataset comes from different distributions. Mismatch can not be prevented by simply increasing the training dataset like we did to overfitting. More information on mismatch will be provided in Homework 11.

## Preparation 2: Local Minima & Saddle Points

### Optimization Failure

<img src="assets/image-20240504115754108.png" alt="image-20240504115754108" style="zoom:25%;" />

<img src="assets/image-20240504115906505.png" alt="image-20240504115906505" style="zoom:25%;" />

Optimization fails not always because of we stuck at local minima. We may also encounter **saddle points**, which are not local minima but have a gradient of $0$. 

All the points that have a gradient of $0$ are called **critical points**. So, we can't say that our gradient descent algorithms always stops because we stuck at local minima -- we may stuck at saddle points as well. The correct way to say is that gradient descent stops when we stuck at a critical point.

If we are stuck at a local minima, then there's no way to further decrease the loss (all the points around local minima are higher); if we are stuck at a saddle point, we can escape the saddle point. But, how can we differentiate a saddle point and local minima?

### Identify which kinds of Critical Points

$L(\boldsymbol{\theta})$ around $\boldsymbol{\theta} = \boldsymbol{\theta}'$ can be approximated (Taylor Series)below:

$$
L(\boldsymbol{\theta}) \approx L(\boldsymbol{\theta}') + (\boldsymbol{\theta} - \boldsymbol{\theta}')^T \boldsymbol{g} + \frac{1}{2} (\boldsymbol{\theta} - \boldsymbol{\theta}')^T H (\boldsymbol{\theta} - \boldsymbol{\theta}')
$$

Gradient $\boldsymbol{g}$ is a *vector*:

$$
\boldsymbol{g} = \nabla L(\boldsymbol{\theta}')
$$

$$
\boldsymbol{g}_i = \frac{\partial L(\boldsymbol{\theta}')}{\partial \boldsymbol{\theta}_i}
$$

Hessian $H$ is a matrix:

$$
H_{ij} = \frac{\partial^2}{\partial \boldsymbol{\theta}_i \partial \boldsymbol{\theta}_j} L(\boldsymbol{\theta}')
$$

<img src="assets/image-20240504121739774.png" alt="image-20240504121739774" style="zoom:25%;" />

The green part is the Gradient and the red part is the Hessian.

When we are at the critical point, The approximation is "dominated" by the Hessian term.

<img src="assets/image-20240504122010303.png" alt="image-20240504122010303" style="zoom:25%;" />

Namely, our approximation formula becomes:

$$
L(\boldsymbol{\theta}) \approx L(\boldsymbol{\theta}') + \frac{1}{2} (\boldsymbol{\theta} - \boldsymbol{\theta}')^T H (\boldsymbol{\theta} - \boldsymbol{\theta}') = L(\boldsymbol{\theta}') + \frac{1}{2} \boldsymbol{v}^T H \boldsymbol{v}
$$

Local minima: 

- For all $\boldsymbol{v}$, if $\boldsymbol{v}^T H \boldsymbol{v} > 0$ ($H$ is positive definite, so all eigenvalues are positive), around $\boldsymbol{\theta}'$: $L(\boldsymbol{\theta}) > L(\boldsymbol{\theta}')$​

Local maxima:

- For all $\boldsymbol{v}$, if $\boldsymbol{v}^T H \boldsymbol{v} < 0$ ($H$ is negative definite, so all eigenvalues are negative), around $\boldsymbol{\theta}'$: $L(\boldsymbol{\theta}) < L(\boldsymbol{\theta}')$

Saddle point:

- Sometimes $\boldsymbol{v}^T H \boldsymbol{v} < 0$, sometimes $\boldsymbol{v}^T H \boldsymbol{v} > 0$. Namely, $H$​ is indefinite -- some eigenvalues are positive and some eigenvalues are negative.

Example:

<img src="assets/image-20240504130115656.png" alt="image-20240504130115656" style="zoom:25%;" />

<img src="assets/image-20240504130345730.png" alt="image-20240504130345730" style="zoom:33%;" />

### Escaping saddle point

If by analyzing $H$'s properpty, we realize that it's indefinite (we are at a saddle point). We can also analyze $H$ to get a sense of the **parameter update direction**! 

Suppose $\boldsymbol{u}$ is an eigenvector of $H$ and $\lambda$ is the eigenvalue of $\boldsymbol{u}$.

$$
\boldsymbol{u}^T H \boldsymbol{u} = \boldsymbol{u}^T (H \boldsymbol{u}) = \boldsymbol{u}^T (\lambda \boldsymbol{u}) = \lambda (\boldsymbol{u}^T \boldsymbol{u}) = \lambda \|\boldsymbol{u}\|^2
$$

If the eigenvalue $\lambda < 0$, then $\boldsymbol{u}^T H \boldsymbol{u} = \lambda \|\boldsymbol{u}\|^2 < 0$ (eigenvector $\boldsymbol{u}$ can't be $\boldsymbol{0}$). Because $L(\boldsymbol{\theta}) \approx L(\boldsymbol{\theta}') + \frac{1}{2} \boldsymbol{u}^T H \boldsymbol{u}$, we know $L(\boldsymbol{\theta}) < L(\boldsymbol{\theta}')$. By definition, $\boldsymbol{\theta} - \boldsymbol{\theta}' = \boldsymbol{u}$. If we perform $\boldsymbol{\theta} = \boldsymbol{\theta}' + \boldsymbol{u}$, we can effectively decrease $L$. We can escape the saddle point and decrease the loss.

However, this method is seldom used in practice because of the huge computation need to compute the Hessian matrix and the eigenvectors/eigenvalues.

### Local minima v.s. saddle point

<img src="assets/image-20240504143925130.png" alt="image-20240504143925130" style="zoom:25%;" />

A local minima in lower-dimensional space may be a saddle point in a higher-dimension space. Empirically, when we have lots of parameters, **local minima is very rare**. 

<img src="assets/image-20240504144357680.png" alt="image-20240504144357680" style="zoom:25%;" />

## Preparation 3: Batch & Momentum

### Small Batch v.s. Large Batch

<img src="assets/image-20240504145118594.png" alt="image-20240504145118594" style="zoom:25%;" />

<img src="assets/image-20240504145348568.png" alt="image-20240504145348568" style="zoom:25%;" />

Note that here, "time for cooldown" does not always determine the time it takes to complete an epoch.

Emprically, large batch size $B$​ does **not** require longer time to compute gradient because of GPU's parallel computing, unless the batch size is too big.

<img src="assets/image-20240504145814623.png" alt="image-20240504145814623" style="zoom:25%;" />

**Smaller** batch requires **longer** time for <u>one epoch</u> (longer time for seeing all data once).

<img src="assets/image-20240504150139906.png" alt="image-20240504150139906" style="zoom:33%;" />

However, large batches are not always better than small batches. That is, the noise brought by small batches lead to better performance (optimization). 

<img src="assets/image-20240504152856857.png" alt="image-20240504152856857" style="zoom:33%;" />

<img src="assets/image-20240504151712866.png" alt="image-20240504151712866" style="zoom: 33%;" />

Small batch is also better on **testing** data (***overfitting***).

![image-20240504154312760](assets/image-20240504154312760.png)

This may be because that large batch is more likely to lead to us stucking at a **sharp minima**, which is not good for testing loss. Because of noises, small batch is more likely to help us escape sharp minima. Instead, at convergence, we will more likely end up in a **flat minima**.

<img src="assets/image-20240504154631454.png" alt="image-20240504154631454" style="zoom: 25%;" />

Batch size is another hyperparameter.

<img src="assets/image-20240504154938533.png" alt="image-20240504154938533" style="zoom:25%;" />

### Momentum

Vanilla Gradient Descent:

<img src="assets/image-20240504160206707.png" alt="image-20240504160206707" style="zoom:25%;" />

Gradient Descent with Momentum:

<img src="assets/image-20240504160436876.png" alt="image-20240504160436876" style="zoom: 25%;" />

<img src="assets/image-20240504160549105.png" alt="image-20240504160549105" style="zoom:25%;" />

### Concluding Remarks

<img src="assets/image-20240504160755845.png" alt="image-20240504160755845" style="zoom:33%;" />

## Preparation 4: Learning Rate

### Problems with Gradient Descent

The fact that training process is stuck does not always mean small gradient.

<img src="assets/image-20240504161124291.png" alt="image-20240504161124291" style="zoom:33%;" />

Training can be difficult even without critical points. Gradient descent can fail to send us to the global minima even under the circumstance of a **convex** error surface. You can't fix this problem by adjusting the learning rate $\eta$.

<img src="assets/image-20240504161746667.png" alt="image-20240504161746667" style="zoom:33%;" />

Learning rate can not be one-size-fits-all. **If we are at a place where the gradient is high (steep surface), we expect $\eta$ to be small so that we don't overstep; if we are at a place where the gradient is small (flat surface), we expect $\eta$ to be large so that we don't get stuck at one place.**

<img src="assets/image-20240504162232403.png" alt="image-20240504162232403" style="zoom:25%;" />

### Adagrad

Formulation for one parameter:

$$
\boldsymbol{\theta}_i^{t+1} \leftarrow \boldsymbol{\theta}_i^{t} - \eta \boldsymbol{g}_i^t
$$

$$
\boldsymbol{g}_i^t = \frac{\partial L}{\partial \boldsymbol{\theta}_i} \bigg |_{\boldsymbol{\theta} = \boldsymbol{\theta}^t}
$$

The new formulation becomes:

$$
\boldsymbol{\theta}_i^{t+1} \leftarrow \boldsymbol{\theta}_i^{t} - \frac{\eta}{\sigma_i^t} \boldsymbol{g}_i^t
$$

$\sigma_i^t$ is both parameter-dependent ($i$) and iteration-dependent ($t$). It is called **Root Mean Square**. It is used in **Adagrad** algorithm.

$$
\sigma_i^t = \sqrt{\frac{1}{t+1} \sum_{i=0}^t (\boldsymbol{g}_i^t)^2}
$$
<img src="assets/image-20240504212350040.png" alt="image-20240504212350040" style="zoom:25%;" />

Why this formulation works?

<img src="assets/image-20240504212744865.png" alt="image-20240504212744865" style="zoom:25%;" />

### RMSProp

However, this formulation still has some problems. We assumed that the gradient for one parameter will stay relatively the same. However, it's not always the case. For example, there may be places where the gradient becomes large and places where the gradient becomes small (as seen from the graph below). The reaction of this formulation to a new gradient change is very slow.

<img src="assets/image-20240504213324493.png" alt="image-20240504213324493" style="zoom:25%;" />

The new formulation is now:

$$
\sigma_i^t = \sqrt{\alpha(\sigma_i^{t-1})^2 + (1-\alpha)(\boldsymbol{g}_i^t)^2}
$$
$\alpha$ is a hyperparameter ($0 < \alpha < 1$). It controls how important the previously-calculated gradient is.

<img src="assets/image-20240504214302296.png" alt="image-20240504214302296" style="zoom:25%;" />

<img src="assets/image-20240504214445048.png" alt="image-20240504214445048" style="zoom:25%;" />

### Adam

The Adam optimizer is basically the combination of RMSProp and Momentum.

![image-20240504214928718](assets/image-20240504214928718.png)

### Learning Rate Scheduling

This is the optimization process with Adagrad:

<img src="assets/image-20240504215606600.png" alt="image-20240504215606600" style="zoom:33%;" />

To prevent the osciallations at the final stage, we can use two methods:

$$
\boldsymbol{\theta}_i^{t+1} \leftarrow \boldsymbol{\theta}_i^{t} - \frac{\eta^t}{\sigma_i^t} \boldsymbol{g}_i^t
$$

#### Learning Rate Decay

As the training goes, we are closer to the destination. So, we reduce the learning rate $\eta^t$​.

<img src="assets/image-20240504220358590.png" alt="image-20240504220358590" style="zoom:25%;" />

This improves the previous result:

<img src="assets/image-20240504220244480.png" alt="image-20240504220244480" style="zoom:33%;" />

#### Warm up

<img src="assets/image-20240504220517645.png" alt="image-20240504220517645" style="zoom:25%;" />

We first increase $\eta ^ t$ and then decrease it. This method is used in both the Residual Network and Transformer paper. At the beginning, the estimate of $\sigma_i^t$​​ has large variance. We can learn more about this method in the RAdam paper.

### Summary

<img src="assets/image-20240504222113767.png" alt="image-20240504222113767" style="zoom:25%;" />

## Preparation 5: Loss

### How to represent classification

We can't directly apply regression to classification problems because regression tends to penalize the examples that are "too correct." 

<img src="assets/image-20240505103637180.png" alt="image-20240505103637180" style="zoom:25%;" />

It's also problematic to directly represent Class 1 as numeric value $1$, Class 2 as $2$, Class 3 as $3$​. That is, this representation has an underlying assumption that Class 1 is "closer" or more "similar" to Class 2 than Class 3. However, this is not always the case.

One possible model is:

$$
f(x) = \begin{cases} 
1 & g(x) > 0 \\
2 & \text{else}
\end{cases} 
$$

The loss function denotes the number of times $f$ gets incorrect results on training data.

$$
L(f) = \sum_n \delta(f(x^n) \neq \hat{y}^n)
$$

We can represent classes as one-hot vectors. For example, we can represent Class $1$ as $\hat{y} = \begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix}$, Class $2$ as $\hat{y} = \begin{bmatrix}
0 \\
1 \\
0
\end{bmatrix}$ and Class $3$ as $\hat{y} = \begin{bmatrix}
0 \\
0 \\
1
\end{bmatrix}$.

<img src="assets/image-20240505084900542.png" alt="image-20240505084900542" style="zoom: 25%;" />

### Softmax

$$
y_i' = \frac{\exp(y_i)}{\sum_j \exp(y_j)}
$$

We know that $0 < y_i' < 1$ and $\sum_i y_i' = 1$.

<img src="assets/image-20240505085254461.png" alt="image-20240505085254461" style="zoom: 33%;" />

<img src="assets/image-20240505085830849.png" alt="image-20240505085830849" style="zoom: 33%;" />

### Loss of Classification

#### Mean Squared Error (MSE)

$$
e = \sum_i (\boldsymbol{\hat{y}}_i - \boldsymbol{y}_i')^2
$$

#### Cross-Entropy

$$
e = -\sum_i \boldsymbol{\hat{y}}_i \ln{\boldsymbol{y}_i'}
$$

Minimizing cross-entropy is equivalent to maximizing likelihood. 

Cross-entropy is more frequently used for classification than MSE. At the region with higher loss, the gradient of MSE is close to $0$. This is not good for gradient descent. 

<img src="assets/image-20240505091600454.png" alt="image-20240505091600454" style="zoom:33%;" />

### Generative Models

<img src="assets/image-20240505110347099.png" alt="image-20240505110347099" style="zoom:25%;" />

$$
P(C_1 \mid x) 
= \frac{P(C_1, x)}{P(x)} 
= \frac{P(x \mid C_1)P(C_1)}{P(x \mid C_1)P(C_1) + P(x \mid C_2)P(C_2)}
$$

We can therefore predict the distribution of $x$:

$$
P(x) = P(x \mid C_1)P(C_1) + P(x \mid C_2)P(C_2)
$$

#### Prior

$P(C_1)$ and $P(C_2)$ are called prior probabilities. 

#### Gaussian distribution

$$
f_{\mu, \Sigma}(x) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu)\right)
$$

Input: vector $x$, output: probability of sampling $x$. The shape of the function determines by mean $\mu$ and covariance matrix $\Sigma$. ==Technically, the output is the probability density, not exactly the probability, through they are positively correlated.==

<img src="assets/image-20240505111630135.png" alt="image-20240505111630135" style="zoom:25%;" />

<img src="assets/image-20240505111652217.png" alt="image-20240505111652217" style="zoom:25%;" />

#### Maximum Likelihood

We assume $x^1, x^2, x^3, \cdots, x^{79}$ generate from the Gaussian ($\mu^*, \Sigma^*$) with the *maximum likelihood*.

$$
L(\mu, \Sigma) = f_{\mu, \Sigma}(x^1) f_{\mu, \Sigma}(x^2) f_{\mu, \Sigma}(x^3) \cdots f_{\mu, \Sigma}(x^{79})
$$

$$
\mu^*, \Sigma^* = \arg \max_{\mu,\Sigma} L(\mu, \Sigma)
$$

The solution is as follows:

$$
\mu^* = \frac{1}{79} \sum_{n=1}^{79} x^n
$$

$$
\Sigma^* = \frac{1}{79} \sum_{n=1}^{79} (x^n - \mu^*)(x^n - \mu^*)^T
$$

<img src="assets/image-20240505115811655.png" alt="image-20240505115811655" style="zoom:25%;" />

But the above generative model fails to give a high-accuracy result. Why? In that formulation, every class has its unique mean vector and covariance matrix. The size of the covariance matrix tends to increase as the feature size of the input increases. This increases the number of trainable parameters, which tends to result in overfitting. Therefore, we can force different distributions to **share the same covariance matrix**.

<img src="assets/image-20240505120606165.png" alt="image-20240505120606165" style="zoom:25%;" />

<img src="assets/image-20240505121134979.png" alt="image-20240505121134979" style="zoom:25%;" />

Intuitively, the new covariance matrix is the sum of the original covariance matrices weighted by the frequencies of samples in each distribution.

<img src="assets/image-20240505121610514.png" alt="image-20240505121610514" style="zoom: 33%;" />

<img src="assets/image-20240505122313657.png" alt="image-20240505122313657" style="zoom: 25%;" />

#### Three steps to a probability distribution model

<img src="assets/image-20240505123718777.png" alt="image-20240505123718777" style="zoom: 25%;" />

We can always use whatever distribution we like (we use Guassian in the previous example).

If we assume all the dimensions are independent, then you are using **Naive Bayes Classifier**.

$$
P(\boldsymbol{x} \mid C_1) =
P(\begin{bmatrix}x_1 \\ x_2 \\ \vdots \\ x_K \end{bmatrix} \mid C_1) =
P(x_1 \mid C_1)P(x_2 \mid C_1) \dots P(x_K \mid C_1)
$$

Each $P(x_m \mid C_1)$ is now a 1-D Gaussian. For binary features, you may assume they are from Bernouli distributions. 

But if the assumption does not hold, the Naive Bayes Classifier may have a very high bias.

#### Posterior Probability

$$
\begin{align}
P(C_1 | x) 
&= \frac{P(x | C_1) P(C_1)}{P(x | C_1) P(C_1) + P(x | C_2) P(C_2)} \\ 
&= \frac{1}{1 + \frac{P(x | C_2) P(C_2)}{P(x | C_1) P(C_1)}} \\
&= \frac{1}{1 + \exp(-z)} \\ 
&= \sigma(z) \\
\end{align}
$$

$$
\begin{align}
z &= \ln \frac{P(x | C_1) P(C_1)}{P(x | C_2) P(C_2)} \\
&= \ln \frac{P(x | C_1)}{P(x | C_2)} + \ln \frac{P(C_1)}{P(C_2)} \\
&= \ln \frac{P(x | C_1)}{P(x | C_2)} + \ln \frac{\frac{N_1}{N_1+N_2}}{\frac{N_2}{N_1+N_2}} \\
&= \ln \frac{P(x | C_1)}{P(x | C_2)} + \ln \frac{N_1}{N_2} \\
\end{align}
$$

Furthermore:

$$
\begin{align}
\ln \frac{P(x | C_1)}{P(x | C_2)}
&= \ln \frac{\frac{1}{(2\pi)^{D/2} |\Sigma_1|^{1/2}} \exp\left\{-\frac{1}{2} (x - \mu^1)^T \Sigma_1^{-1} (x - \mu^1)\right\}}  {\frac{1}{(2\pi)^{D/2} |\Sigma_2|^{1/2}} \exp\left\{-\frac{1}{2} (x - \mu^2)^T \Sigma_2^{-1} (x - \mu^2)\right\}} \\
&= \ln \frac{|\Sigma_2|^{1/2}}{|\Sigma_1|^{1/2}} \exp \left\{ -\frac{1}{2} [(x - \mu^1)^T \Sigma_1^{-1} (x - \mu^1)-\frac{1}{2} (x - \mu^2)^T \Sigma_2^{-1} (x - \mu^2)] \right\} \\
&= \ln \frac{|\Sigma_2|^{1/2}}{|\Sigma_1|^{1/2}} - \frac{1}{2} \left[(x - \mu^1)^T \Sigma_1^{-1} (x - \mu^1) - (x - \mu^2)^T \Sigma_2^{-1} (x - \mu^2)\right]
\end{align}
$$

Further simplification goes:

<img src="assets/image-20240505132500740.png" alt="image-20240505132500740" style="zoom:33%;" />

Since we assume the distributions share the covariance matrix, we can further simplify the formula:

<img src="assets/image-20240505133107451.png" alt="image-20240505133107451" style="zoom:33%;" />

$$
P(C_1 \mid x) = \sigma(w^Tx + b)
$$

This is why the decision boundary is a linear line.

In generative models, we estimate $N_1, N_2, \mu^1, \mu^2, \Sigma$, then we have $\boldsymbol{w}$ and $b$. How about directly find $\boldsymbol{w}$ and $b$​?

### Logistic Regression

We want to find $P_{w,b}(C_1 \mid x)$. If $P_{w,b}(C_1 \mid x) \geq 0.5$, output $C_1$. Otherwise, output $C_2$.

$$
P_{w,b}(C_1 \mid x) = \sigma(z) = \sigma(w \cdot x + b) 
= \sigma(\sum_i w_ix_i + b)
$$

The function set is therefore (including all different $w$ and $b$):

$$
f_{w,b}(x) = P_{w,b}(C_1 \mid x)
$$

Given the training data $\{(x^1, C_1),(x^2, C_1),(x^3, C_2),\dots, (x^N, C_1)\}$, assume the data is generated based on $f_{w,b}(x) = P_{w,b}(C_1 \mid x)$. Given a set of $w$ and $b$, the probability of generating the data is:

$$
L(w,b) = f_{w,b}(x^1)f_{w,b}(x^2)\left(1-f_{w,b}(x^3)\right)...f_{w,b}(x^N)
$$

$$
w^*,b^* = \arg \max_{w,b} L(w,b)
$$

We can write the formulation by introducing $\hat{y}^i$, where:

$$
\hat{y}^i = \begin{cases}
1 & x^i \text{ belongs to } C_1 \\
0 & x^i \text{ belongs to } C_2
\end{cases}
$$

<img src="assets/image-20240505153535990.png" alt="image-20240505153535990" style="zoom:33%;" />

<img src="assets/image-20240505153917703.png" alt="image-20240505153917703" style="zoom:33%;" />

$$
C(p,q) = - \sum_x p(x) \ln \left( q(x) \right)
$$

Therefore, minimizing $- \ln L(w,b)$ is actually minimizing the cross entropy between two distributions: the output of function $f_{w,b}$ and the target $\hat{y}^n$​​.

$$
L(f) = \sum_n C(f(x^n), \hat{y}^n)
$$

$$
C(f(x^n), \hat{y}^n) = -[\hat{y}^n \ln f(x^n) + (1-\hat{y}^n) \ln \left(1-f(x^n)\right)]
$$

<img src="assets/image-20240505155715704.png" alt="image-20240505155715704" style="zoom:33%;" />

<img src="assets/image-20240505155812019.png" alt="image-20240505155812019" style="zoom:33%;" />

<img src="assets/image-20240505160902451.png" alt="image-20240505160902451" style="zoom:33%;" />

Here, the larger the difference ($\hat{y}^n - f_{w,b}(x^n)$) is, the larger the update.

Therefore, the update step for **logistic regression** is:

$$
w_i \leftarrow w_i - \eta \sum_n - \left(\hat{y}^n - f_{w,b}(x^n)\right)x_i^n
$$

This looks the same as the update step for linear regression. However, in logistic regression, $f_{w,b}, \hat{y}^n \in \{0,1\}$.

Comparision of the two algorithms:

<img src="assets/image-20240505161330795.png" alt="image-20240505161330795" style="zoom: 25%;" />

Why using square error instead of cross entropy on logistic regression is a bad idea?

<img src="assets/image-20240505163118191.png" alt="image-20240505163118191" style="zoom:25%;" />

<img src="assets/image-20240505163307888.png" alt="image-20240505163307888" style="zoom:25%;" />

In either case, this algorithm fails to produce effective optimization. A visualization of the loss functions for both cross entropy and square error is illustrated below:

<img src="assets/image-20240505163520499.png" alt="image-20240505163520499" style="zoom:25%;" />

### Discriminative v.s. Generative

The logistic regression is an example of **discriminative** model, while the Gaussian posterior probability method is an example of **generative** model, through their function set is the same.

<img src="assets/image-20240505170417654.png" alt="image-20240505170417654" style="zoom:25%;" />

We will not obtain the same set of $w$ and $b$. The same model (function set) but different function is selected by the same training data. The discriminative model tends to have a better performance than the generative model.

A toy example shows why the generative model tends to perform less well. We assume Naive Bayes here, namely $P(x \mid C_i) = P(x_1 \mid C_i)P(x_2 \mid C_i)$ if $x \in \mathbb{R}^2$. The result is counterintuitive -- we expect the testing data to be classified as Class 1 instead of Class 2.

<img src="assets/image-20240505202709608.png" alt="image-20240505202709608" style="zoom:25%;" />

<img src="assets/image-20240505211619095.png" alt="image-20240505211619095" style="zoom:25%;" />

### Multiclass Classification

<img src="assets/image-20240505213248614.png" alt="image-20240505213248614" style="zoom:33%;" />

**Softmax** will further enhance the maximum $z$ input, expanding the difference between a large value and a small value. Softmax is an approximation of the posterior probability. If we assume the previous Gaussian generative model that share the same covariance matrix amongst distributions, we can derive the exact same Softmax formulation. We can also derive Softmax from maximum entropy (similar to logistic regression).

<img src="assets/image-20240505213741874.png" alt="image-20240505213741874" style="zoom: 25%;" />

Like the binary classification case earlier, the multiclass classification aims to maximize likelihood, which is the same as minimizing cross entropy.

### Limitations of Logistic Regression

<img src="assets/image-20240505220032474.png" alt="image-20240505220032474" style="zoom: 25%;" />

Solution: **feature transformation**

<img src="assets/image-20240505220341723.png" alt="image-20240505220341723" style="zoom:25%;" />

However, it is *not* always easy to find a good transformation. We can **cascade logistic regression models**.

<img src="assets/image-20240505220557595.png" alt="image-20240505220557595" style="zoom: 25%;" />

<img src="assets/image-20240505220908418.png" alt="image-20240505220908418" style="zoom:25%;" />

# 3/04 Lecture 3: Image as input

## Preparation 1: CNN

### CNN

A neuron does not have to see the whole image. Every receptive field has a set of neurons (e.g. 64 neurons).

<img src="assets/image-20240506100008130.png" alt="image-20240506100008130" style="zoom:25%;" />

The same patterns appear in different regions. Every receptive field has the neurons with the same set of parameters.

<img src="assets/image-20240506100525093.png" alt="image-20240506100525093" style="zoom:25%;" />

<img src="assets/image-20240506101006793.png" alt="image-20240506101006793" style="zoom:25%;" />

The convolutional layer produces a **feature map**.

<img src="assets/image-20240506103951132.png" alt="image-20240506103951132" style="zoom:25%;" />

<img src="assets/image-20240506104022627.png" alt="image-20240506104022627" style="zoom:25%;" />

The feature map becomes a "new image." Each filter convolves over the input image.

A filter of size 3x3 will not cause the problem of missing "large patterns." This is because that as we go deeper into the network, our filter will read broader information (as seen from the illustration below).

<img src="assets/image-20240506104431235.png" alt="image-20240506104431235" style="zoom:25%;" />

Subsampling the pixels will not change the object. Therefore, we always apply **Max Pooling** (or other methods) after convolution to reduce the computation cost.

<img src="assets/image-20240506123931836.png" alt="image-20240506123931836" style="zoom:25%;" />

<img src="assets/image-20240506123958219.png" alt="image-20240506123958219" style="zoom:25%;" />

<img src="assets/image-20240506124013529.png" alt="image-20240506124013529" style="zoom:25%;" />

### Limitations of CNN

CNN is *not* invariant to **scaling and rotation** (we need **data augumentation**).

# 3/11 Lecture 4: Sequence as input

## Preparation 1: Self-Attention

### Vector Set as Input

We often use **word embedding** for sentences. Each word is a vector, and therefore the whole sentence is a vector set.

<img src="assets/image-20240506131746897.png" alt="image-20240506131746897" style="zoom:25%;" />

Audio can also be represented as a vector set. We often use a vector to represent a $25ms$-long audio.

<img src="assets/image-20240506131950596.png" alt="image-20240506131950596" style="zoom:25%;" />

Graph is also a set of vectors (consider each **node** as a *vector* of various feature dimensions).

### Output

Each vector has a label (e.g. POS tagging). This is also called **sequence labeling**.

<img src="assets/image-20240506132918462.png" alt="image-20240506132918462" style="zoom:25%;" />

The whole sequence has a label (e.g. sentiment analysis).

<img src="assets/image-20240506132941134.png" alt="image-20240506132941134" style="zoom:25%;" />

It is also possible that the model has to decide the number of labels itself (e.g. translation). This is also called **seq2seq**.

<img src="assets/image-20240506133219680.png" alt="image-20240506133219680" style="zoom:25%;" />

### Sequence Labeling

<img src="assets/image-20240506142936680.png" alt="image-20240506142936680" style="zoom: 25%;" />

The **self-attention** module will try to consider the whole sequence and find the relevant vectors within the sequence (based on the attention score $\alpha$ for each pair).

<img src="assets/image-20240506143433074.png" alt="image-20240506143433074" style="zoom:25%;" />

There are many ways to calculate $\alpha$:

**Additive**:

<img src="assets/image-20240506143747993.png" alt="image-20240506143747993" style="zoom:25%;" />

**Dot-product**: (the most popular method)

<img src="assets/image-20240506143624304.png" alt="image-20240506143624304" style="zoom:25%;" />

<img src="assets/image-20240506144202322.png" alt="image-20240506144202322" style="zoom: 33%;" />

The attention score will then pass through *softmax* (not necessary, RELU is also possible).

$$
\alpha_{1,i}' = \frac{\exp(\alpha_{1,i})}{\sum_j \exp(\alpha_{1,j})}
$$

<img src="assets/image-20240506144352946.png" alt="image-20240506144352946" style="zoom: 33%;" />

We will then extract information based on attention scores (after applying softmax).

$$
\boldsymbol{b}^1 = \sum_i \alpha_{1,i}' \boldsymbol{v}^i
$$

<img src="assets/image-20240506144754754.png" alt="image-20240506144754754" style="zoom: 33%;" />

If $\boldsymbol{a}^1$ is most similar to $\boldsymbol{a}^2$, then $\alpha_{1,2}'$ will be the highest. Therefore, $\boldsymbol{b}^1$ will be dominated by $\boldsymbol{a}^2$.

## Preparation 2: Self-Attention

### Review

The creation of $\boldsymbol{b}^n$ is in parallel. We don't wait.

<img src="assets/image-20240506145502633.png" alt="image-20240506145502633" style="zoom:25%;" />

### Matrix Form

We can also view self-attention using matrix algebra.

Since every $\boldsymbol{a}^n$ will produce $\boldsymbol{q}^n, \boldsymbol{k}^n, \boldsymbol{v}^n$​, we can write the process in matrix-matrix multiplication form. 

<img src="assets/image-20240506152837314.png" alt="image-20240506152837314" style="zoom:25%;" />

Remeber that:

<img src="assets/image-20240506152558882.png" alt="image-20240506152558882" style="zoom: 50%;" />

As a result, we can write:

<img src="assets/image-20240506152818613.png" alt="image-20240506152818613" style="zoom:33%;" />

In addition, we can use the same method for calculating attention scores:

<img src="assets/image-20240506153556204.png" alt="image-20240506153556204" style="zoom:33%;" />

Here, since $K = [\boldsymbol{k}^1, \boldsymbol{k}^2, \boldsymbol{k}^3, \boldsymbol{k}^4]$​, we use its transpose $K^T$.

By applying softmax, we make sure that every column of $A'$ sum up to $1$, namely, for $i\in\{1,2,3,4\}$, $\sum_j \alpha_{i,j}' = 1$​​.

We use the same method to write the final output $\boldsymbol{b}^n$:

<img src="assets/image-20240506155151917.png" alt="image-20240506155151917" style="zoom:33%;" />

This is based on matrix-vector rules:

<img src="assets/image-20240506155319537.png" alt="image-20240506155319537" style="zoom: 50%;" />

Summary of self-attention: the process from $I$ to $O$

<img src="assets/image-20240506155530554.png" alt="image-20240506155530554" style="zoom:33%;" />

### Multi-Head Self-Attention

We may have different metrics of relevance. As a result, we may consider multi-head self-attention, a variant of self-attention.

<img src="assets/image-20240506161132825.png" alt="image-20240506161132825" style="zoom:33%;" />

We can then combine $\boldsymbol{b}^{i,1}, \boldsymbol{b}^{i,2}$ to get the final $\boldsymbol{b}^i$.

<img src="assets/image-20240506161448289.png" alt="image-20240506161448289" style="zoom: 33%;" />

### Positional Encoding

Self-attention does not care about position information. For example, it does not care whether $\boldsymbol{a}^1$ is close to $\boldsymbol{a}^2$ or $\boldsymbol{a}^4$. To solve that, we can apply **positional encoding**. Each position has a unique **hand-crafted** positional vector $\boldsymbol{e}^i$. We then apply $\boldsymbol{a}^i \leftarrow \boldsymbol{a}^i + \boldsymbol{e}^i$​​.

<img src="assets/image-20240506163634996.png" alt="image-20240506163634996" style="zoom:25%;" />

### Self-Attention for Speech

If the input sequence is of length $L$, the attention matrix $A'$ is a matrix of $L$x$L$​, which may require a large amount of computation. Therefore, in practice, we don't look at the whole audio sequence. Instead, we use **truncated self-attention**, which only looks at a small range.

<img src="assets/image-20240506163652228.png" alt="image-20240506163652228" style="zoom:25%;" />

### Self-Attention for Images

<img src="assets/image-20240506164243254.png" alt="image-20240506164243254" style="zoom:25%;" />

What's its difference with CNN? 

- CNN is the self-attention that can only attends in a receptive field. Self-attention is a CNN with learnable receptive field.
- CNN is simplified self-attention. Self-attention is the complex version of CNN.

<img src="assets/image-20240506165326461.png" alt="image-20240506165326461" style="zoom:25%;" />

Self-attention is more flexible and therefore more prune to overfitting if the dataset is not large enough. We can also use **conformer**, a combination of the two.

<img src="assets/image-20240506165258983.png" alt="image-20240506165258983" style="zoom: 33%;" />

### Self-Attention v.s. RNN

Self-attention is a more complex version of RNN. RNN can be bi-directional, so it is possible to consider the whole input sequence like self-attention. However, it struggles at keeping a vector at the start of the sequence in the memory. It's also computationally-expensive because of its sequential (non-parallel) nature.

<img src="assets/image-20240506170101038.png" alt="image-20240506170101038" style="zoom:33%;" />

### Self-Attention for Graphs

<img src="assets/image-20240506170517327.png" alt="image-20240506170517327" style="zoom:33%;" />

This is one type of GNN.

# 3/18 Lecture 5: Sequence to sequence

## Preparation 1 & 2: Transformer

### Roles

We input a sequence and the model output a sequence. The output length is determined by the model. We use it for speech recognition and translation.

<img src="assets/image-20240506172314513.png" alt="image-20240506172314513" style="zoom:25%;" />

Seq2seq is widely used for QA tasks. Most NLP applications can be viewed as QA tasks. However, we oftentimes use specialized models for different NLP applications.

<img src="assets/image-20240506175405073.png" alt="image-20240506175405073" style="zoom:25%;" />

Seq2seq can also be used on **multi-label classification** (an object can belong to *multiple* classes). This is different from **multi-class classification**, in which we need to classify an object into *one* class out of many classes.

<img src="assets/image-20240506214852300.png" alt="image-20240506214852300" style="zoom:25%;" />

The basic components of seq2seq is:

<img src="assets/image-20240506223508365.png" alt="image-20240506223508365" style="zoom:25%;" />

### Encoder

We need to output a sequence that has the same length as the input sequence. We can technically use CNN or RNN to accomplish this. In Transformer, they use self-attention.

<img src="assets/image-20240506223712032.png" alt="image-20240506223712032" style="zoom:25%;" />

The state-of-the-art encoder architecture looks like this:

<img src="assets/image-20240506224019755.png" alt="image-20240506224019755" style="zoom:25%;" />

The Transformer architecture looks like this:

<img src="assets/image-20240506224349208.png" alt="image-20240506224349208" style="zoom:33%;" />

Residual connection is a very popular technique in deep learning: $\text{output}_{\text{final}} = \text{output} + \text{input}$

<img src="assets/image-20240506225128224.png" alt="image-20240506225128224" style="zoom:33%;" />

### Decoder

#### Autoregressive (AT)

<img src="assets/image-20240507095254276.png" alt="image-20240507095254276" style="zoom:33%;" />

Decoder will receive input that is the its own output in the last timestamp. If the decoder made a mistake in the last timestamp, it will continue that mistake. This may cause **error propagation**.

<img src="assets/image-20240507095529003.png" alt="image-20240507095529003" style="zoom:28%;" />

The encoder and the decoder of the Transformer is actually quite similar if we hide one part of the decoder.

<img src="assets/image-20240507095905602.png" alt="image-20240507095905602" style="zoom:38%;" />

**Masked self-attention**: When considering $\boldsymbol{b}^i$, we will only take into account $\boldsymbol{k}^j$, $j \in [0,i)$. This is because the decoder does not read the input sequence all at once. Instead, the input token is generated one after another.

<img src="assets/image-20240507100340717.png" alt="image-20240507100340717" style="zoom:36%;" />

We also want to add a **stop token** (along with the vocabulary and the start token) to give the decoder a mechanism that it can control the length of the output sequence.

#### Non-autoregressive (NAT)

<img src="assets/image-20240507110111305.png" alt="image-20240507110111305" style="zoom:25%;" />

How to decide the output length for NAT decoder? 

- Another predictor for output length.
- Determine a maximum possible length of sequence, $n$. Feed the decoder with $n$​ START tokens. Output a very long sequence, ignore tokens after END.

Advantage: **parallel** (relying on self-attention), **more stable generation** (e.g., TTS) -- we can control the output-length classifier to manage the length of output sequence

NAT is usually *worse* than AT because of multi-modality.

### Encoder-Decoder

Cross Attention:

<img src="assets/image-20240507111202594.png" alt="image-20240507111202594" style="zoom:33%;" />

$\alpha_i'$ is the attention score after softmax:

<img src="assets/image-20240507111524741.png" alt="image-20240507111524741" style="zoom:33%;" />

### Training

<img src="assets/image-20240507112444947.png" alt="image-20240507112444947" style="zoom:33%;" />

This is very similar to how to we train a **classification** model. Every time the model creates an output, the model makes a classification.

Our goal is to minimize the sum of cross entropy of all the outputs.

<img src="assets/image-20240507121645401.png" alt="image-20240507121645401" style="zoom:33%;" />

**Teacher forcing**: using the ground truth as input.

#### Copy Mechanism 

Sometimes we may want the model to just copy the input token. For example, consider a *chatbot*, when the user inputs "*Hi, I'm ___*," we don't expect the model to generate an output of the user's name because it's not likely in our training set. Instead, we want the model to learn the pattern: *when it sees input "I'm [some name]*," it can output "*Hi [some name]. Nice to meet you!*"

#### Guided Attention 

In some tasks, input and output are monotonically aligned.
For example, speech recognition, TTS (text-to-speech), etc. We don't want the model to miass some important portions of the input sequence in those tasks. 

<img src="assets/image-20240507155629681.png" alt="image-20240507155629681" style="zoom:33%;" />

We want to force the model to learn a particular order of attention.

- monotonic attention
- location-aware attention

#### Beam Search

The red path is **greedy decoding**. However, if we give up a little bit at the start, we may get a better global optimal path. In this case, the green path is the best one.

<img src="assets/image-20240507160226512.png" alt="image-20240507160226512" style="zoom:33%;" />

We can use beam search to find a heuristic.

However, sometimes **randomness** is needed for decoder when generating sequence in some tasks (e.g. sentence completion, TTS). In those tasks, finding a "good path" may not be the best thing because there's no correct answer and we want the model to be "creative." In contrast, beam search may be more beneficial to tasks like speech recognition.

#### Scheduled Sampling

At training, the model always sees the "ground truth" -- correct input sequence. At testing, the model is fed with its own output in the previous round. This may cause the model to underperform because it may never see a "wrong input sequence" before (**exposure bias**). Therefore, we may want to train the model with some wrong input sequences. This is called **Scheduled Sampling**. But this may hurt the parallel capability of the Transformer.

# 3/25 Lecture 6: Generation

## Preparation 1: GAN Basic Concepts

<img src="assets/image-20240507163642230.png" alt="image-20240507163642230" style="zoom:25%;" />

Generator is a network that can output a distribution.

Why we bother add a distribution into our network?

In this video game frame prediction example, the model is trained on a dataset of two coexisting possibilities -- the role turning left and right. As a result, the vanilla network will seek to balance the two. Therefore, it could create a frame that a role splits into two: one turning left and one turning right.

<img src="assets/image-20240507164409222.png" alt="image-20240507164409222" style="zoom:25%;" />

This causes problems. Therefore, we want to add a distribution into the network. By doing so, the output of the network will also become a distribution itself. We especially prefer this type of network when our tasks need "creativity." The same input has different correct outputs.

### Unconditional Generation

For unconditional generation, we don't need the input $x$.

<img src="assets/image-20240507170223915.png" alt="image-20240507170223915" style="zoom:33%;" />

GAN architecture also has a **discriminator**. This is just a vanilla neural network (CNN, transformer ...) we've seen before.

<img src="assets/image-20240507170541953.png" alt="image-20240507170541953" style="zoom: 25%;" />

The *adversarial* process of GAN looks like this:

<img src="assets/image-20240507172617903.png" alt="image-20240507172617903" style="zoom:25%;" />

The algorithm is:

1. (Randomly) initialize generator and discriminator's parameters
2. In each training iteration:
   1. **Fix generator $G$ and update discriminator $D$**. This task can be seen as either a classification (labeling true images as $1$ and generator-generated images $0$​) or regression problem. We want discriminator to **learn to assign high scores to real objects and local scores to generated objects**.
   2. **Fix discriminator $D$ and update generator $G$**. Generator learns to "fool" the discriminator. We can use **gradient ascent** to train the generator while freezing the paramters of the discriminator.

<img src="assets/image-20240507180402008.png" alt="image-20240507180402008" style="zoom:30%;" />

<img src="assets/image-20240507180417153.png" alt="image-20240507180417153" style="zoom:30%;" />

The GNN can also learn different angles of face. For example, when we apply **interpolation** on one vector that represents a face facing left and the other vector that represents a face to the right. If we feed the resulting vector into the model, the model is able to generate a face to the middle.

<img src="assets/image-20240507182338648.png" alt="image-20240507182338648" style="zoom: 20%;" />

## Preparation 2: Theory Behind GAN

<img src="assets/image-20240507183702031.png" alt="image-20240507183702031" style="zoom:30%;" />

$$
G^* = \arg \min_G Div(P_G, P_{\text{data}})
$$

where $Div(P_G, P_{\text{data}})$, our "loss function," is the **divergence** between two distributions: $P_G$ and $P_{\text{data}}$.

The hardest part of GNN training is how to formulate the divergence. But, sampling is good enough. Although we do not know the distributions of $P_G$ and $P_{\text{data}}$, we can sample from them.

<img src="assets/image-20240507185732875.png" alt="image-20240507185732875" style="zoom:25%;" />

For discriminator, 

<img src="assets/image-20240507190414858.png" alt="image-20240507190414858" style="zoom:33%;" />

$$
D^* = \arg \max_D V(D,G)
$$

$$
V(G, D) = \mathbb{E}_{y \sim P_{\text{data}}} [\log D(y)] + \mathbb{E}_{y \sim P_G} [\log (1 - D(y))]
$$

Since we want to maximize $V(G,D)$​, we in turn wants the discriminator output for true data to be as large as possible and the discriminator output for generated output to be as small as possible.

Recall that cross-entropy $e = -\sum_i \boldsymbol{\hat{y}}_i \ln{\boldsymbol{y}_i'}$. <u>We can see that $V(G,D)$ looks a lot like **negative cross entropy** $-e = \sum_i \boldsymbol{\hat{y}}_i \ln{\boldsymbol{y}_i'}$.</u>

Since we often minimize cross-entropy, we can find similarities here as well: $\min e = \max -e = \max V(G,D)$​. As a result, when we do the above optimization on a discriminator, we are actually training a *classifier* (with cross-entropy loss). That is, we can **view a discriminator as a classifier** that tries to seperate the true data and the generated data.

In additon, $\max_D V(D,G)$ is also related to **JS divergence** (proof is in the original GAN paper):

<img src="assets/image-20240507191902403.png" alt="image-20240507191902403" style="zoom:33%;" />

Therefore,

$$
\begin{align}
G^* &= \arg \min_G Div(P_G, P_{\text{data}}) \\
&= \arg \min_G \max_D V(D,G)
\end{align}
$$

This is how the GAN algorithm was designed (to solve the optimization problem above).

GAN is known for its difficulty to be trained. 

**In most cases, $P_G$ and $P_{\text{data}}$ are not overlapped.** 

- The nature of the data is that both $P_G$ and $P_{\text{data}}$​ are **low-dimensional manifold in a high-dimensional space**. That is, most pictures in the high-dimensional space are not pictures, let alone human faces. So, any overlap can be ignored.

- Even when $P_G$ and $P_{\text{data}}$ have overlap, the discriminator could still divide them if we don't have enough sampling.

  <img src="assets/image-20240507201344462.png" alt="image-20240507201344462" style="zoom:25%;" />

The problem with JS divergence is that JS divergence always outputs $\log2$ if two distributions do not overlap.

<img src="assets/image-20240507201618857.png" alt="image-20240507201618857" style="zoom:25%;" />

In addition, **when two classifiers don't overlap, binary classifiers can always achieve $100\%$ accuracy**. Everytime we finish discriminator training, the accuracy is $100\%$. We had hoped that after iterations, the discriminator will struggle more with classifying true data from generated data. However, it's not the case -- our discriminator can always achieve $100\%$ accuracy.

The accuracy (or loss) means nothing during GAN training.

#### WGAN

<img src="assets/image-20240507203904606.png" alt="image-20240507203904606" style="zoom:25%;" />

Considering one distribution P as a pile of earth, and another distribution Q as the target, the **Wasserstein Distance** is the average distance the earth mover has to move the earth. In the case above, distribution $P$ is concentrated on one point. Therefore, the distance is just $d$​.

However, when we consider two distributions, the distance can be difficult to calculate.

<img src="assets/image-20240507204341296.png" alt="image-20240507204341296" style="zoom:28%;" />

Since there are many possible "moving plans," we use the “moving plan” with the **smallest** average distance to define the Wasserstein distance.

$W$ is a better metric than $JS$ since it can better capture the divergence of two distributions with no overlap.

<img src="assets/image-20240507204742569.png" alt="image-20240507204742569" style="zoom:25%;" />

$$
W(P_{\text{data}}, P_G) = \max_{D \in \text{1-Lipschitz}} \left\{ \mathbb{E}_{y \sim P_{\text{data}}} [D(y)] - \mathbb{E}_{y \sim P_{G}} [D(y)] \right\}
$$

$D \in \text{1-Lipschitz}$ means that $D(x)$ has to be a smooth enough function. Having this constraint prevents $D(x)$ from becoming $\infty$ and $-\infty$.

<img src="assets/image-20240507222633511.png" alt="image-20240507222633511" style="zoom:33%;" />

When the two distributions are very close, the two extremes can't be too far apart. This causes $W(P_{\text{data}}, P_G)$ to become relatively small. When the two distributions are very far, the two extremes can be rather far apart, making $W(P_{\text{data}}, P_G)$ relatively large.
