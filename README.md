***Acknowledgement***:

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
w^*, b^* = \arg\min_{w,b} L
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
y = c\cdot \frac{1}{1 + e^{-(b + wx_1)}} = c\cdot \text{sigmoid}(b + wx_1)
$$

We can change $w$, $c$ and $b$ to get sigmoid curves with different shapes.

Different Sigmoid curves -> Combine to approximate different piecewise linear functions -> Approximate different continuous functions

<img src="assets/image-20240503124800696.png" alt="image-20240503124800696" style="zoom: 25%;" />

<img src="assets/image-20240503125224141.png" alt="image-20240503125224141" style="zoom: 25%;" />

From a model with high bias $y=b+wx_1$ to the new model with more features and a much lower bias:

$$
y = b + \sum_{i} c_i \cdot \text{sigmoid}(b_i + w_i x_1)
$$

Also, if we consider multiple features $y = b + \sum_{j} w_j x_j$​, the new model can be expanded to look like this:

$$
y = b + \sum_{i} c_i \cdot \text{sigmoid}(b_i + \sum_{j} w_{ij} x_j)
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
2. calculate the gradient $\boldsymbol{g} = \begin{bmatrix} \frac{\partial{L}} {\partial{\theta_1}}\bigg|_{\boldsymbol{\theta}=\boldsymbol{\theta}^0} \\ \frac{\partial{L}}{\partial{\theta_2}}\bigg|_{\boldsymbol{\theta}=\boldsymbol{\theta}^0} \\ \vdots \end{bmatrix} = \nabla L(\boldsymbol{\theta}^0)$ with $\boldsymbol{\theta}, \boldsymbol{g} \in \mathbb{R}^n$
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

<img src="assets/image-20240526124823314.png" alt="image-20240526124823314" style="zoom:33%;" />

Large loss doesn't always imply issues with model bias. There may be issues with *optimization*. That is, gradient descent does not always produce global minima. We may stuck at a local minima. In the language of function set, the set theoretically contain optimal function $f^*(\boldsymbol{x})$. However, we may never reach that.

<img src="assets/image-20240526124848542.png" alt="image-20240526124848542" style="zoom:30%;" />

### Optimization issue

- gain insights from comparison to identify whether the failure of the model is due to optimization issues, overfitting or model bias
- start from shallower networks (or other models like SVM which is much easier to optimize)
- if deeper networks do not contain smaller loss on *training* data,  then there is optimization issues (as seen from the graph below)

<img src="assets/image-20240504100720536.png" alt="image-20240504100720536" style="zoom:25%;" />

For example, here, the 5-layer should always do better or the same as the 4-layer network. This is clearly due to optimization problems.

<img src="assets/image-20240526125018446.png" alt="image-20240526125018446" style="zoom:33%;" />

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

<img src="assets/image-20240526124935031.png" alt="image-20240526124935031" style="zoom:25%;" />

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
f(x) = \begin{cases} 1 & g(x) > 0 \\ 2 & \text{else} \end{cases}
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
P(C_1 \mid x) = \frac{P(C_1, x)}{P(x)} = \frac{P(x \mid C_1)P(C_1)}{P(x \mid C_1)P(C_1) + P(x \mid C_2)P(C_2)}
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

## Preparation 6: Batch Normalization

<img src="assets/image-20240508173240007.png" alt="image-20240508173240007" style="zoom:33%;" />

In a linear model, when the value at each dimension have very distinct values, we may witness an error surface that is steep in one dimension and smooth in the other, as seen from the left graph below. If we can restrict the value at each dimension to be of the same range, we can make the error surface more "trainable," as seen from the right graph below.

<img src="assets/image-20240509091139135.png" alt="image-20240509091139135" style="zoom:30%;" />

#### Feature Normalization

Recall **standardization**:
$$
\boldsymbol{\tilde{x}}^{r}_{i} \leftarrow \frac{\boldsymbol{x}^{r}_{i} - m_{i}}{\sigma_{i}}
$$
$i$ represents the dimension of the vector $\boldsymbol{x}$ and $r$ represents the index of the datapoint.

<img src="assets/image-20240509092317165.png" alt="image-20240509092317165" style="zoom:25%;" />

For the *Sigmoid* activation function, we can apply feature normalization on $\boldsymbol{z}$ (before Sigmoid) so that all the values are concentrated close to $0$. But in other cases, it's acceptable to apply feature normalization on either $\boldsymbol{z}$ or $\boldsymbol{a}$.

<img src="assets/image-20240509094310111.png" alt="image-20240509094310111" style="zoom:33%;" />

When doing feature normalization, we can use element-wise operation on $\boldsymbol{z}$.

<img src="assets/image-20240509094746248.png" alt="image-20240509094746248" style="zoom:29%;" />

#### Batch Normalization Training

Notice that now **if $\boldsymbol{z}^1$ changes, $\boldsymbol{\mu}, \boldsymbol{\sigma}, \boldsymbol{\tilde{z}}^1, \boldsymbol{\tilde{z}}^2, \boldsymbol{\tilde{z}}^3$​ will all change**. That is, the network is now considering all the inputs and output a bunch of outputs. This could be slow and memory-intensive because we need to load the entire dataset. So, we consider batches -- *Batch Normalization*.

<img src="assets/image-20240509100535438.png" alt="image-20240509100535438" style="zoom:33%;" />

We can also make a small improvement:

<img src="assets/image-20240509101408124.png" alt="image-20240509101408124" style="zoom:33%;" />
$$
\boldsymbol{\hat{z}}^{i} = \boldsymbol{\gamma} \odot \boldsymbol{\tilde{z}}^{i} + \boldsymbol{\beta}
$$
We set $\boldsymbol{\gamma}$ to $[1,1,...]^T$ and $\boldsymbol{\beta}$ to $[0,0,...]^T$​ at the start of the iteration (they are *parameters* of the network). This means that the range will still the the same amongst different dimensions in the beginning. As the training goes, we may want to lift the contraint that each dimension has a mean of 0.

#### Batch Normalization Testing/Inference

<img src="assets/image-20240509102533136.png" alt="image-20240509102533136" style="zoom:33%;" />

We do not always have batch at testing stage. 

Computing the moving average of $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$​ of the batches during training (PyTorch has an implementation of this). The hyperparamteer $p$ is usually $0.1$. $\boldsymbol{\mu^t}$ is the $t$-th **batch**'s $\boldsymbol{\mu}$. 
$$
\boldsymbol{\bar{\mu}} \leftarrow 
p \boldsymbol{\bar{\mu}} + (1-p) \boldsymbol{\mu^t}
$$


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

### Speech

If the input sequence is of length $L$, the attention matrix $A'$ is a matrix of $L$x$L$​, which may require a large amount of computation. Therefore, in practice, we don't look at the whole audio sequence. Instead, we use **truncated self-attention**, which only looks at a small range.

<img src="assets/image-20240506163652228.png" alt="image-20240506163652228" style="zoom:25%;" />

### Images

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

### Graphs

<img src="assets/image-20240506170517327.png" alt="image-20240506170517327" style="zoom:33%;" />

This is one type of GNN.

## Extra Material: RNN

How to represent a word?

### 1-of-N Encoding

The vector is lexicon size. Each dimension corresponds to a word in the lexicon. The dimension for the word is 1, and others are 0. For example, lexicon = {apple, bag, cat, dog, elephant}. Then, apple = [1 0 0 0 0].

<img src="assets/image-20240509112046739.png" alt="image-20240509112046739" style="zoom:25%;" />

### RNN architecture

<img src="assets/image-20240509112132338.png" alt="image-20240509112132338" style="zoom:25%;" />

The memory units must have an initial value. **Changing the order of the sequence will change the output**.

<img src="assets/image-20240509112231949.png" alt="image-20240509112231949" style="zoom:25%;" />

**The same network is used again and again**. If the words before a particular word is different, then the values stored in the memory will be different, therefore causing the probability (i.e. output) of the same word different.

We can also make the RNN deeper:

<img src="assets/image-20240509121949403.png" alt="image-20240509121949403" style="zoom:25%;" />

### Elman & Jordan Network

<img src="assets/image-20240509122024968.png" alt="image-20240509122024968" style="zoom:25%;" />

Jordan Network tends to have a *better* performance because we can know what exactly is in the memory unit (the output itself).

### Bidirectional RNN

<img src="assets/image-20240509122605942.png" alt="image-20240509122605942" style="zoom:33%;" />

We can also train two networks at once. This way, the network can consider the entire input sequence.

### Long Short-Term Memory (LSTM)

<img src="assets/image-20240509123050221.png" alt="image-20240509123050221" style="zoom:33%;" />

$4$ inputs: input, signal to Input Gate, signal to Output Gate, signal to Forget Gate

Why LSTM? The RNN will wipe out memory in every new timestamp, therefore having a really short short-term memory. However, the LSTM can hold on to the memory as long as the Forget Gate $f(z_f)$ is $1$​.

This is the structure of a **LSTM cell/neuron**:

<img src="assets/image-20240509124021283.png" alt="image-20240509124021283" style="zoom:40%;" />

When $f(z_f)=1$, memory $c$ is completely remembered; when $f(z_f)=0$, memory $c$ is completely forgotten (since $c \cdot f(z_f)=0$​).

When $f(z_i)=1$, input $g(z)$ is completely passed through; when $f(z_i)=0$, $g(z)$ is blocked.

Same story for $f(z_o)$.

### Multi-Layer LSTM

A LSTM neuron has **four** times more parameters than a vanilla neural network neuron. In vanilla neural network, every neuron is a function mapping a input vector to a output scalar. In LSTM, the neuron maps $4$ inputs to $1$ output.

<img src="assets/image-20240509131055385.png" alt="image-20240509131055385" style="zoom:38%;" />

Assume the total number of cells is $n$.

<img src="assets/image-20240509131711965.png" alt="image-20240509131711965" style="zoom:40%;" />

We will apply $4$ linear transformations to get $\boldsymbol{z^f, z^i, z, z^o}$. Each of them represents a type of input to the LSTM neuron. Each of them is a vector in $\mathbb{R}^n$. The $i$-th entry is the input to the $i$​​​-th neuron.

We can use element-wise operations on those vectors to conduct these operations for all $n$ cells at the same time. In addition, the $z$ inputs are based on not only the current input $\boldsymbol{x^t}$, but also the memory cell $\boldsymbol{c^{t-1}}$ and the previous output $\boldsymbol{h^{t-1}}$.

<img src="assets/image-20240509134617159.png" alt="image-20240509134617159" style="zoom:40%;" />

### RNN Training

RNN training relies on Backpropagation Through Time (BPTT), a variant of backpropagation. RNN-based network is not always easy to learn.

<img src="assets/image-20240509141000994.png" alt="image-20240509141000994" style="zoom:33%;" />

This is why it is difficult to train RNN. When we are at the flat plane, we often have a large learning rate. If we happen to step on the edge of the steep cliff, we may jump really far (because of the high learning rate and high gradient). This may cause optimization failure and segmentation fault.

This can be solved using **Clipping**. We can set a maximum threshold for the gradient, so that it does not become really high.

<img src="assets/image-20240509141022472.png" alt="image-20240509141022472" style="zoom:33%;" />

Why do we observe this kind of behavior for RNN's error surface?

<img src="assets/image-20240509144606662.png" alt="image-20240509144606662" style="zoom:33%;" />

We can notice that the same weight $w$ is applied many times in different time. This causes the problem because any change in $w$ will either has not effect on the final output $y^N$ or a huge impact.

One solution is LSTM. It can deal with **gradient vanishing**, but not **gradient explode**. As a result, most points on LSTM error surface with have a high gradient. When training LSTM, we can thus set the learning rate a relatively small value.

Why LSTM solves gradient vanishing? Memory and input are added. **The influence never disappears unless Forget Gate is closed**. 

An alternative option is Gated Recurrent Unit (GRU). It is simpler than LSTM. It only has 2 gates: combining the Forget Gate and the Input Gate.

### More Applications

RNN can also be applied on **many-to-one** tasks (input is a vector sequence, but output is only one output), such as sentiment analysis. We can set the output of RNN at the last timestamp as our final output.

RNN can be used on many-to-many tasks (both input and output are vector sequences, but the output is shorter). For example, when doing speech recognition task, we can use **Connectionist Temporal Classification (CTC)**.

<img src="assets/image-20240509153641320.png" alt="image-20240509153641320" style="zoom: 25%;" />

RNN can also be applied on sequence-to-sequence learning (both input and output are both sequences with different lengths), such as machine translation.

<img src="assets/image-20240509154402105.png" alt="image-20240509154402105" style="zoom: 25%;" />

## Extra Material: GNN

How do we utilize the structures and relationship to help train our model?

### Spatial-Based Convolution

<img src="assets/image-20240509170309414.png" alt="image-20240509170309414" style="zoom:33%;" />

- **Aggregation**: use neighbor features to update hidden states in the next layer
- **Readout**: use features of all the nodes to represent the whole graph $h_G$​

#### NN4G (Neural Network for Graph)

<img src="assets/image-20240509170939928.png" alt="image-20240509170939928" style="zoom:35%;" />
$$
h_3^0 = \boldsymbol{w}_0 \cdot \boldsymbol{x}_3
$$

$$
h_3^1 = \hat{w}_{1,3}(h_0^0 + h_2^0 + h_4^0) + \boldsymbol{w}_1 \cdot \boldsymbol{x}_3
$$

<img src="assets/image-20240509171000501.png" alt="image-20240509171000501" style="zoom:35%;" />

#### DCNN (Diffusion-Convolution Neural Network)

<img src="assets/image-20240509174350496.png" alt="image-20240509174350496" style="zoom:33%;" />
$$
h_3^0 = w_3^0 MEAN(d(3,\cdot)=1)
$$

$$
h_3^1 = w_3^1 MEAN(d(3,\cdot)=2)
$$

Node features:

<img src="assets/image-20240509174728965.png" alt="image-20240509174728965" style="zoom:33%;" />



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

**Residual connection** is a very popular technique in deep learning: $\text{output}_{\text{final}} = \text{output} + \text{input}$

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

When the two distributions are very close, the two extremes can't be too far apart. This causes $W(P_{\text{data}}, P_G)$ to become relatively small. When the two distributions are very far, the two extremes can be rather far apart, making $W(P_{\text{data}}, P_G)$​​ relatively large.

## Preparation 3: Generator Performance and Conditional Generation

<img src="assets/image-20240508084957535.png" alt="image-20240508084957535" style="zoom:25%;" />

GAN is still challenging because if either generator or decoder fails to imrpove, the other will fail.

More tips on how to train GAN:

<img src="assets/image-20240508085939200.png" alt="image-20240508085939200" style="zoom:25%;" />

### Sequence Generation GAN

GAN can also be applied on *sequence generation*:

In this case, the seq2seq model becomes our generator. However, this can be very hard to train. Since a tiny change in the parameter of generator will likely not affect the output of the generator (since the output is the most likely one), the score stays unchanged.

<img src="assets/image-20240508090236679.png" alt="image-20240508090236679" style="zoom:33%;" />

As a result, you can not use gradient descent on this. Therefore, we usually apply RL on Sequence Generation GAN.

Usually, the generator are fine-tuned from a model learned by other approaches. However, with enough hyperparameter-tuning and tips, **ScarchGAN** can train from scratch. 

There are also other types of generative models: VAE and Flow-Based Model.

### Supervised Learning on GAN

There are some other possible solutions. We can assign a vector to every image in the training set. We can then train the model using the vector and image pair.

<img src="assets/image-20240508091644143.png" alt="image-20240508091644143" style="zoom:25%;" />

### Evaluation of Generation

Early in the start of GAN, we rely on human evaluation to judge the performance of generation. However, human evaluation is expensive (and sometimes unfair/unstable). How to evaluate the quality of the generated images automatically? 

We can use an image classifer to solve this problem:

<img src="assets/image-20240508092339349.png" alt="image-20240508092339349" style="zoom:33%;" />

If the generated image is not like a real image, the classifer will have a more evenly spread distribution.

#### Mode Collapse

However, the generator could still be subject to a problem called **Mode Collapse** (a diversity problem). After generating more and more images, you could observe that the generated images look mostly the same.

<img src="assets/image-20240508100403689.png" alt="image-20240508100403689" style="zoom:33%;" />

This could be because the generator can learn from the discriminator's weakness and focus on that particular weakness.

#### Mode Dropping

<img src="assets/image-20240508101011624.png" alt="image-20240508101011624" style="zoom:33%;" />

**Mode Dropping** is more difficult to be detected. The distribution of generated data is diverse but it's still a portion of the real distribution.

#### Inception Score (IS)

Good quality and large diversity will produce a larger IS.

<img src="assets/image-20240508101614367.png" alt="image-20240508101614367" style="zoom: 33%;" />

<img src="assets/image-20240508101641023.png" alt="image-20240508101641023" style="zoom:33%;" />

#### Fréchet Inception Distance (FID)

<img src="assets/image-20240508102617665.png" alt="image-20240508102617665" style="zoom: 33%;" />

All the points are the direct results (before being passed into softmax) produced by CNN. We also feed real images into the CNN to produce the red points. We assume both blue and red points come from *Gaussian* distributions. This may sometimes be problematic. In addition, to accurately obtain the distributions, we may need a lot of samples, which may lead to a huge computation cost. 

However, we don't want a "memory GAN." If the GAN just outputs the samples (or alter them a tiny amount, like flipping), it may actually obtain a very low FID score. However, this is not what we want.

### Conditional Generation

Condition generation is useful for text-to-image tasks.

<img src="assets/image-20240508105411552.png" alt="image-20240508105411552" style="zoom:33%;" />

The discriminator will look at two things:

- is output image $y$ realistic or not
- are text input $x$ and $y$ matched or not

<img src="assets/image-20240508121623002.png" alt="image-20240508121623002" style="zoom:33%;" />

It's important that we add a good image which does not match the text input as our training data.

It's also common to apply Conditional GAN on **image translation**, i.e. **pix2pix**.

<img src="assets/image-20240508122011547.png" alt="image-20240508122011547" style="zoom:33%;" />

We can technically use either supervised learning or GAN to design the model. However, in terms of supervised learning, since there're many correct outputs for a given input, the model may try to fit to all of them. The best output is when we use both GAN and supervised learning.

<img src="assets/image-20240508122344067.png" alt="image-20240508122344067" style="zoom:30%;" />

Conditional GAN can also be applied on sound-to-image generation.

## Preparation 4: Cycle GAN

GAN can also be applied on unsupervised learning. In some cases like **Image Style Transfer**, we may not be able to obtain any paired data. We could still learn mapping from unpaired data using **Unsupervised Conditional Generation**.

<img src="assets/image-20240508165308900.png" alt="image-20240508165308900" style="zoom:25%;" />

Instead of sampling from a Gaussian distribution like vanilla GAN, we sample from a particular domain $\mathcal{X}$. In this case, the domain is human profiles.

<img src="assets/image-20240508165636665.png" alt="image-20240508165636665" style="zoom:33%;" />

However, the model may try to ignore the input because as long as it generates something from domain $\mathcal{Y}$, it can pass the discriminator check. We can use the **Cycle GAN** architecture: training two generators at once. In this way, $G_{\mathcal{X} \rightarrow \mathcal{Y}}$ has to generate something related to the "input", so that $G_{\mathcal{Y} \rightarrow \mathcal{X}}$ can reconstruct the image.

<img src="assets/image-20240508170522970.png" alt="image-20240508170522970" style="zoom:33%;" />

In theory, the two generators may learn some strange mappings that prevent the model from actually output a related image. However, in practice, even with vanilla GAN, image style transfer can be learned by the model (the model prefers simple mapping, i.e. output something that looks like the input image).

Cycle GAN can also be in both ways:

<img src="assets/image-20240508171406589.png" alt="image-20240508171406589" style="zoom:33%;" />

It can also be applied on **text-style transfer**. This idea is also applied on Unsupervised Abstractive Summarization, Unsupervised Translation, Unsupervised ASR.

# 4/01 Recent Advance of Self-supervised learning for NLP

## Self-supervised Learning

Self-supervised learning is a form of unsupervised learning.

### Masking Input

<img src="assets/image-20240509183930967.png" alt="image-20240509183930967" style="zoom:30%;" />

BERT is a transformer encoder, outputing a sequence of the same length as the input sequence.

We randomly mask some tokens with either a special token or some random tokens. We then aim to train the model to minimize the cross entropy between the output and the ground truth. During training, we aim to update parameters of both the BERT and the linear model.

### Next Sentence Prediction

<img src="assets/image-20240509184440678.png" alt="image-20240509184440678" style="zoom:30%;" />

This task tries to train the model to output a "yes" for thinking that sentence 2 does follow sentence 1 and a "no" for thinking that sentence 2 does not follow sentence 1. However, this approach is not very useful probably because this task is relatively easy and the model does not learn very much. An alternative way is to use **SOP**, which trains the model to learn whether sentence 1 is before sentence 2 or after sentence 2.

### Fine-tuning

With self-supervised learning, we **pre-train** the model, which prepares the model for **downstream tasks** (tasks we care and tasks that we've prepared a little bit labeled data). We can then **fine-tune** the model for different tasks.

With a combination of unlabelled and labelled dataset, this training is called **semi-supervised learning**.

#### GLUE

GLUE has $9$ tasks. We fine-tune $9$ models based on the fine-tuned BERT, each for one task below, and then evaluate the model's performance.

<img src="assets/image-20240509190900208.png" alt="image-20240509190900208" style="zoom:25%;" />

#### Sequence to Class

Example: sentiment analysis

<img src="assets/image-20240509191651068.png" alt="image-20240509191651068" style="zoom:30%;" />

Note that here, we only randomly initialize the parameters in the linear model, instead of BERT. Also, it's important that we train the model with labelled dataset.

Fining tuning the model can accelerate the optimization process and improve the performance (lower the loss), as shown in the graph below.

<img src="assets/image-20240509191956464.png" alt="image-20240509191956464" style="zoom:33%;" />

#### Sequence to Sequence (Same Length)

Example: POS tagging

<img src="assets/image-20240509193758857.png" alt="image-20240509193758857" style="zoom:30%;" />

#### Two Sequences to Class

<img src="assets/image-20240509211910633.png" alt="image-20240509211910633" style="zoom:30%;" />

Example: **Natural Language Inference (NLI)**

<img src="assets/image-20240509212007348.png" alt="image-20240509212007348" style="zoom:30%;" />

#### Extraction-Based Question Answering (QA)

<img src="assets/image-20240509212306860.png" alt="image-20240509212306860" style="zoom:32%;" />

In this case, as usual, both $D$ and $Q$ are sequences of vectors. Both $d_i$ and $q_i$ are words (vectors). In QA system, $D$ is the article and $Q$ is the question. The outputs are two positive integers $s$ and $e$. This means that the answer is in the range from the $s$-th word to the $e$-th word.

<img src="assets/image-20240509213240963.png" alt="image-20240509213240963" style="zoom:30%;" />

<img src="assets/image-20240509213311833.png" alt="image-20240509213311833" style="zoom:30%;" />

The orange and blue vectors are the only two set of parameters we need to train.

### Pre-Train Seq2seq Models

It's also possible to pre-train a seq2seq model. We can corrupt the input to the Encoder and let the Decoder reconstruct the original input.

<img src="assets/image-20240509214652739.png" alt="image-20240509214652739" style="zoom:30%;" />

There're many ways you can use to corrupt the input.

<img src="assets/image-20240509214809925.png" alt="image-20240509214809925" style="zoom:28%;" />

### Why Does BERT work?

The tokens with similar meaning have similar embedding. BERT can output token vectors that represent similar words with high cosine similarity. 

<img src="assets/image-20240510130519076.png" alt="image-20240510130519076" style="zoom:35%;" />

There was a similar work conducted called **CBOW word embedding**. Using deep learning, BERT appeas to perform better. It can learn the **contextualized word embedding**.

However, learning context of a word may not be the whole story. You can even use BERT to classify meaningless sentences. For example, you can replace parts of a DNA sequence with a random word so that a whole DNA sequence becomes a sentence. You can even use BERT to classify the DNA sequences represented with sentences. Similar works have been done on **protein, DNA and music classification**.

### Multi-Lingual BERT

BERT can also be trained on many languages.

**Cross-Lingual Alignment**: assign similar embeddings to the same words from different languages.

**Mean Reciprocal Rank (MRR)**: Higher MRR, better alignment 

### GPT

GPT uses a different pre-training method -- next token prediction.

# 4/22 Lecture 8: Auto-encoder / Anomaly Detection

## Auto-encoder

Auto-encoder is often used for **dimension reduction** (the output vector of the NN encoder). This idea is very similar to **Cycle GAN**.

<img src="assets/image-20240510141903138.png" alt="image-20240510141903138" style="zoom:33%;" />

**The variation of pictures is very limited**. There are many possible $n \times n$ matrix. But only a very small subset of them are meaningful pictures. So, the job of the encoder is to make complex things simpler. As seen from the graph below, we can represent the possible pictures with a $1\times2$ vector.

<img src="assets/image-20240510150707632.png" alt="image-20240510150707632" style="zoom:25%;" />

### De-noising Auto-encoder

<img src="assets/image-20240510151357919.png" alt="image-20240510151357919" style="zoom:33%;" />

BERT can actually be seen as a de-nosing auto-encoder. Note that the decoder does not have to be a linear model.

<img src="assets/image-20240510151515147.png" alt="image-20240510151515147" style="zoom:25%;" />

### Feature Disentanglement

The embedding includes information of different aspects.

<img src="assets/image-20240510153514659.png" alt="image-20240510153514659" style="zoom:33%;" />

However, we don't need know exactly what dimensions of an embedding contains a particular aspect of information.

Related papers:

<img src="assets/image-20240510153802051.png" alt="image-20240510153802051" style="zoom:33%;" />

Application: Voice Conversion

<img src="assets/image-20240510161627849.png" alt="image-20240510161627849" style="zoom:25%;" />

### Discrete Latent Representation

<img src="assets/image-20240510161651820.png" alt="image-20240510161651820" style="zoom:33%;" />

In the last example, be constraining the embedding to be one-hot, the encoder is able to learn a classification problem (*unsupervised* learning).

The most famous example is: **Vector Quantized Variational Auto-encoder (VQVAE)**.

<img src="assets/image-20240510162432579.png" alt="image-20240510162432579" style="zoom:33%;" />

The parameters we learn in VQVAE are: Encoder, Decoder, the $5$ vectors in the Codebook

This process is very similar to attention. We can think of the output of the Encoder as the Query, the Codebook as Values and Keys.

This method constrains the input of the Decoder to be one of the inputs in the Codebook.

Using this method on speech, the model can learn phonetic information which is represented by the Codebook.

### Text as Representation

<img src="assets/image-20240510163559946.png" alt="image-20240510163559946" style="zoom:33%;" />

We can achieve unsupervised summarization using the architecture above (actually it's just a CycleGAN model). We only need crawled documents to train the model.

Both the Encoder and the Decoder should use **seq2seq** models because they accept sequence as input and output. The Discriminator makes sure the summary is readable. This is called a **seq2seq2seq auto-encoder**. 

### Generator

We can also gain a generator by using the trained Decoder in auto-encoder.

<img src="assets/image-20240510170518530.png" alt="image-20240510170518530" style="zoom:33%;" />

With some modification, we have variational auto-encoder (VAE).

### Compression

The image reconstruction is not $100$%. So, there may be a "lossy" effect on the final reconstructed image.

<img src="assets/image-20240510170803281.png" alt="image-20240510170803281" style="zoom: 33%;" />

## Anomaly Detection

Anomaly Detection: Given a set of training data $\{x^1, x^2, ..., x^N\}$, detecting input $x$​​ is *similar* to training data or not

<img src="assets/image-20240510183428138.png" alt="image-20240510183428138" style="zoom:28%;" />

Use cases:

<img src="assets/image-20240510183459488.png" alt="image-20240510183459488" style="zoom:33%;" />

Anomaly Detection is *better* than training a binary classifier because we often don't have the dataset of anomaly (oftentimes we humans can't even detect anomaly ourselves). Our dataset is primarily composed of "normal" data. Anomaly Detection is often called one-class classification.

This is how we implement anomaly detecter:

<img src="assets/image-20240510184817485.png" alt="image-20240510184817485" style="zoom:30%;" />

There are many ways we can implement anomaly detection, besides anomaly detecter.

# 4/29 Lecture 9: Explainable AI

## Explainable ML
### Interpretable v.s. Powerful

Some models are intrinsically interpretable.

- For example, linear model (from weights, you know the
  importance of features)
- But not very powerful
- Deep network is difficult to be interpretable. Deep networks are black boxes, but powerful than a linear model. 

For a cat classifier:

**Local Explanation**: why do you think *this image* is a cat?

**Global Explanation**: what does a *cat* look like? (not referring to a specific image)

### Local Explanation

For an object $x$ (image, text, etc), it can be composed of multiple **components**: $\{x_1, ..., x_n, ..., x_N\}$. For each $x_i$ in image it can represent pixel, segment, etc. For each $x_i$ in text, it can represent a word.

We asks the question: "which component is critical for making a decision?" We can therefore **remove or modify the components**. The important components are those that result in **large decision change**.

#### Saliency Map

<img src="assets/image-20240511092757980.png" alt="image-20240511092757980" style="zoom:38%;" />

Sometimes, our Saliency Map may appear noisy (noisy gradient). We can use **SmoothGrad** to solve that -- Randomly *add* noises to the input image, get saliency maps of the noisy images, and *average* them.

<img src="assets/image-20240511092953563.png" alt="image-20240511092953563" style="zoom:33%;" />

#### Integrated Gradient (IG)

However, there may be problems with **gradient saturation**. Gradients can not always reflect importance. In this case, a longer trunk size does not make the object *more* elephant. We can use Integrated Gradient to solve this problem.

<img src="assets/image-20240511095234208.png" alt="image-20240511095234208" style="zoom:33%;" />

#### Network Visualization

<img src="assets/image-20240511100752281.png" alt="image-20240511100752281" style="zoom:33%;" />

We can learn that the network knows that the same sentence spoken by different speaker is similar (erase the identity of the speaker).

#### Network Probing

<img src="assets/image-20240511131445426.png" alt="image-20240511131445426" style="zoom:30%;" />

Using this method, we can tell whether the embeddings have information of POS properties by evaluating the accuracy of the **POS** (Part of Speech) or **NER** (Named-Entity Recognition) classifier. If the accuracy is high, then the embeddings do contain those information.

If the classifier itself has very high bias, then we can not really tell whether the embeddings have those information based on accuracy.

<img src="assets/image-20240511133759883.png" alt="image-20240511133759883" style="zoom:30%;" />

We can also train a TTS model that takes the output of a particular hidden layer as input. If the TTS model outputs an audio sequence that has no speaker information. Then, we can conclude that the model has learned to earse speaker information when doing audio-to-text tasks.

### Global Explanation

<img src="assets/image-20240511133947764.png" alt="image-20240511133947764" style="zoom:33%;" />

We can know based on feature maps (the output of a filter) what patterns a particular filter is responsible for detecting. As a result, for an unknown image $X$, we can try to create an image that tries to create a large value of sum in a feature map:

$$
X^* = \arg \max_X \sum_i \sum_j a_{ij}
$$

$X^*$ is the image that contains the patterns filter 1 can detect. We can find $X^*$ using **gradient *ascent***. One example of the resulting $X^*$ looks like this:

<img src="assets/image-20240511140050243.png" alt="image-20240511140050243" style="zoom:28%;" />

However, if we try to find the best image (that maximizes the classification probability), 

$$
X^* = \arg \max_X y_i
$$

we fail to see any meaningful pattern:

<img src="assets/image-20240511140257814.png" alt="image-20240511140257814" style="zoom:30%;" />

The images should look like a digit. As a result, we need to add constraints $R(X)$ -- how likely $X$ is a digit:

$$
X^* = \arg \max_X y_i + R(X)
$$

where:

$$
R(X) = - \sum_{i,j} |X_{i,j}|
$$

$R(X)$ restricts image $X$ to have as little pattern (white regions) as possible since an image of a digit should have a couple of patterns (the rest of the image should just be the cover).

<img src="assets/image-20240511140732770.png" alt="image-20240511140732770" style="zoom:33%;" />

To make the images clearer, we may need several regularization terms, and hyperparameter tuning... 

For example, we can append an image generator $G$ (GAN, VAE, etc) before the classifier that we want to understand.

<img src="assets/image-20240511140917496.png" alt="image-20240511140917496" style="zoom:35%;" />

$$
z^* = \arg \max_z y_i
$$

We can then get the image we want by calculating:

$$
X^* = G(z^*)
$$

<img src="assets/image-20240511141059751.png" alt="image-20240511141059751" style="zoom:33%;" />

### Outlook

We can use an interpretable model (linear model) to mimic the behavior of an uninterpretable model, e.g. NN (not the complete behavior, only "local range").

<img src="assets/image-20240511142144142.png" alt="image-20240511142144142" style="zoom:33%;" />

This idea is called Local Interpretable Model-Agnostic Explanations (**LIME**)

# 5/06 Lecture 10: Attack

## Adversarial Attack

### Example of Attack

<img src="assets/image-20240511195843588.png" alt="image-20240511195843588" style="zoom:33%;" />

### White Box Attack

Assume we know the parameters $\boldsymbol{\theta}$ of the Network.

<img src="assets/image-20240511195926194.png" alt="image-20240511195926194" style="zoom:33%;" />

Our goal is to find a new picture $\boldsymbol{x}$ that correspond to an output $\boldsymbol{y}$ that is most different from the true label (one-hot) $\boldsymbol{\hat{y}}$. 

Since we also want the noise to be as little as possible, we add an additional constraint: $d(\boldsymbol{x^0}, \boldsymbol{x}) \leq \epsilon$. $\epsilon$​ is a threshold such that we want the noise to be unable to be perceived by humans.

$$
\boldsymbol{x^*} = \arg 
\min_{\boldsymbol{x}, d(\boldsymbol{x^0}, \boldsymbol{x}) \leq \epsilon} 
L(\boldsymbol{x})
$$

We also define $\boldsymbol{\Delta x}$:
$$
\boldsymbol{\Delta x} = \boldsymbol{x} - \boldsymbol{x^0}
$$
This can be seen as:
$$
\begin{bmatrix}
\Delta x_1 \\
\Delta x_2 \\
\Delta x_3 \\
\vdots
\end{bmatrix}
=
\begin{bmatrix}
x_1 \\
x_2 \\
x_3 \\
\vdots
\end{bmatrix}
-
\begin{bmatrix}
x_1^0 \\
x_2^0 \\
x_3^0 \\
\vdots
\end{bmatrix}
$$
There are many ways to calculate distance $d$​ between the two images:

**L2-norm**:
$$
d(\boldsymbol{x^0}, \boldsymbol{x}) =
\|\boldsymbol{\Delta x}\|_2 =
\sqrt{(\Delta x_1)^2 + (\Delta x_2)^2 + \dots}
$$
**L-infinity**:
$$
d(\boldsymbol{x^0}, \boldsymbol{x}) =
\|\boldsymbol{\Delta x}\|_{\infty} =
\max\{|\Delta x_1|,|\Delta x_2|, \dots\}
$$
L-infinity is a *better* metric for images because it fits human perception of image differences (as seen from the example below). We may need to use other metrics depending on our domain knowledge.

<img src="assets/image-20240511203532567.png" alt="image-20240511203532567" style="zoom:27%;" />

The Loss function can be defined depending on the specific attack:

**Non-Targeted Attack**:

The Loss function $L$ is defined to be the negative cross entropy between the true label and the output label. Since we want to minimize $L$, this is the same as maximizing $ e(\boldsymbol{y}, \boldsymbol{\hat{y}}) $.

$$
L(\boldsymbol{x}) = - e(\boldsymbol{y}, \boldsymbol{\hat{y}})
$$

**Targeted Attack:**

Since we want to *minimize* the Loss function, we add the cross entropy between the output and the target label (since we want the two to be similar, therefore a smaller $e$).

$$
L(\boldsymbol{x}) = - e(\boldsymbol{y}, \boldsymbol{\hat{y}}) + 
e(\boldsymbol{y}, \boldsymbol{y^{\text{target}}})
$$

How can we do optimization?

If we assume we don't have the constraint on $\min$, it's the same as the previous NN training, except that we now update *input*, not parameters (because parameters are now constant).

1. Start from the *original* image $\boldsymbol{x^0}$
2. From $t=1$ to $T$: $\boldsymbol{x^t} \leftarrow \boldsymbol{x^{t-1}} - \eta \boldsymbol{g}$​

If we consider the constraint, the process is very similar:

1. Start from the *original* image $\boldsymbol{x^0}$
2. From $t=1$ to $T$: $\boldsymbol{x^t} \leftarrow \boldsymbol{x^{t-1}} - \eta \boldsymbol{g}$​​
3. If $d(\boldsymbol{x^0}, \boldsymbol{x}) > \epsilon$, then $\boldsymbol{x^t} \leftarrow fix(\boldsymbol{x^t})$​

If we are using L-infinity, we can fix $\boldsymbol{x^t}$ by finding a point within the constraint that is the closest to $\boldsymbol{x^t}$.

<img src="assets/image-20240511205802496.png" alt="image-20240511205802496" style="zoom:28%;" />

#### FGSM

We can use **Fast Gradient Sign Method (FGSM, https://arxiv.org/abs/1412.6572)**:

Redefine $\boldsymbol{g}$ as:
$$
\mathbf{g} = \begin{bmatrix} 
\text{sign}\left(\frac{\partial L}{\partial x_1} \bigg|_{\mathbf{x}=\mathbf{x}^{t-1}}\right) \\ 
\text{sign}\left(\frac{\partial L}{\partial x_2} \bigg|_{\mathbf{x}=\mathbf{x}^{t-1}} \right) \\ 
\vdots 
\end{bmatrix}
$$
Here, each entry is either $1$ or $-1$: if $t>0$, $\text{sign}(t) = 1$; otherwise, $\text{sign}(t) = -1$.

1. Start from the *original* image $\boldsymbol{x^0}$
2. Do only one shot: $\boldsymbol{x^t} \leftarrow \boldsymbol{x^{t-1}} - \epsilon \boldsymbol{g}$​​

<img src="assets/image-20240511210950529.png" alt="image-20240511210950529" style="zoom:28%;" />

In the case of L-infinity, $\boldsymbol{x}$ will move to either of the four *corners*.

There's also an *iterative* version of FGSM, which is basically doing **Step 2** $T$​ iterations

### Black Box Attack

What if we don't have parameters of a private network?

If you have the training data of the target network:

1. Train a proxy network yourself
2. Using the proxy network to generate attacked objects

If we don't have training data, we can just the targeted network to generate a lot of data points.

<img src="assets/image-20240511221530022.png" alt="image-20240511221530022" style="zoom:33%;" />

It's also possible to do **One Pixel Attack** (only add noise to one pixel of an image) and **Universal Adversarial Attack** (use one noise signals to attack all the image inputs for a model).

#### Attack in Physical World

- An attacker would need to find perturbations that generalize beyond a single image. 
- Extreme differences between adjacent pixels in the perturbation are unlikely to be accurately captured by cameras. 
- It is desirable to craft perturbations that are comprised mostly of colors reproducible by the printer

<img src="assets/image-20240512085236115.png" alt="image-20240512085236115" style="zoom: 25%;" />

#### Adversarial Reprogramming

<img src="assets/image-20240512085321302.png" alt="image-20240512085321302" style="zoom:28%;" />

#### "Backdoor" in Model

- Attack happens in the training phase
- We need to be careful of open public dataset

<img src="assets/image-20240512085352803.png" alt="image-20240512085352803" style="zoom:30%;" />

### Passive Defense

Passive Defense means that we do not change the parameter of the model.

<img src="assets/image-20240513085222368.png" alt="image-20240513085222368" style="zoom:30%;" />

The overall objective is to apply a *filter* to make the attack signal less powerful.

<img src="assets/image-20240513085504231.png" alt="image-20240513085504231" style="zoom:28%;" />

One way to implement this is that we can apply **Smoothing** on the input image. This will alter the attack signal, making it much less harmful. However, we need to pay attention to its side effect -- it may make the classification confidence lower, though it rarely affects the accuracy.

We can also use image compression and generator to implement this.

However, these methods have some drawbacks. For example, if the attackers know these methods are used as defenses, they can obtain a signal that has taken into account these filters (just treat them as the starting layers of the network). Therefore, we can use **randomization**.

<img src="assets/image-20240513090827186.png" alt="image-20240513090827186" style="zoom:33%;" />

### Proactive Defense

We can use **Adversarial Training** (training a model that is robust to adversarial attack). This method is an example of **Data Augumentation**.

Given training set $\mathcal{X} = \{(\boldsymbol{x^1}, \hat{y}^1), ..., (\boldsymbol{x^N}, \hat{y}^N)\}$:

Using $\mathcal{X}$​ to train your model

1. For $n = 1$ to $N$: Find adversarial input $\boldsymbol{\tilde{x}^n}$ given $\boldsymbol{x^n}$​ by **an attack algorithm**

2. We then have new training data $\mathcal{X}' = \{(\boldsymbol{\tilde{x}^1}, \hat{y}^1), ..., (\boldsymbol{\tilde{x}^N}, \hat{y}^N)\}$

3. Using both $\mathcal{X}$ and $\mathcal{X}'$​​ to update your model
4. Repeat Steps 1-3 

However, if we did not consider an attack algorithm in Adversarial Training, this method will not prevent the attack from that algorithm.

However, Adversarial Training is very computationally *expensive*.

# 5/13 Lecture 11: Adaptation

## Domain Adaptation

<img src="assets/image-20240513093652027.png" alt="image-20240513093652027" style="zoom:25%;" />

**Domain shift**: Training and testing data have different distributions

There are also many different aspects of doman shift:

<img src="assets/image-20240513094020488.png" alt="image-20240513094020488" style="zoom:30%;" />

<img src="assets/image-20240513094038533.png" alt="image-20240513094038533" style="zoom:32%;" />

If we have some labeled data points from the target domain, we can train a model by source data, then *fine-tune* the model by target data. However, since there's only limited target data, we need to be careful about *overfitting*.

However, the most common scenairo is that we often have large amount of unlabeled target data.

<img src="assets/image-20240513095238778.png" alt="image-20240513095238778" style="zoom:33%;" />

We can train a Feature Extractor (a NN) to output the features of the input.

### Domain Adversarial Training

<img src="assets/image-20240513095752543.png" alt="image-20240513095752543" style="zoom:36%;" />

Domain Classifier is a binary classifier. The goal of the Label Predictor is to be as accurate as possible on Source Domain images. The goal of the Domain Classifier is to be as accurate as possible on any image.

### Domain Generalization

If we do not know anything about the target domain, then the problem we are dealing with is Domain Generalization.

# 5/20 Lecture 12: Reinforcement Learning

## RL

Just like ML, RL is also looking for a function, the Actor.

<img src="assets/image-20240513123225764.png" alt="image-20240513123225764" style="zoom:33%;" />

### Policy Network

<img src="assets/image-20240513124558164.png" alt="image-20240513124558164" style="zoom:33%;" />

The **Actor** architecture can be anything: FC NN, CNN, transformer...

Input of NN: the observation of machine represented as a vector or a matrix 

Output of NN: each **action** corresponds to a neuron in output layer (they represent the probability of the action the Actor will take -- we don't always take the action with maximum probability)

### Loss

An **episode** is the whole process from the first observation $s_1$​ to game over.

The **total reward**, a.k.a. **return**, is what we want to maximize:

$$
R = \sum_{t=1}^T r_t
$$

### Optimization

A **trajectory** $\tau$ is the set of all the observations and actions.

$$
\tau = \{s_1, a_1, s_2, a_2, ...\}
$$

Reward is a function of $s_i$ and $a_i$, the current observation and the current action.

<img src="assets/image-20240513125325278.png" alt="image-20240513125325278" style="zoom:35%;" />

This idea is similar to GAN. The Actor is like a Generator; the Reward and the Environment together are like the Discriminator. However, in RL, the Reward and the Environment are not NN. This makes the problem not able to be solved by gradient descent.

### Control an Actor

Make an Actor take or *don't* take a specific action $\hat{a}$ given specific observation $s$.

<img src="assets/image-20240513133923534.png" alt="image-20240513133923534" style="zoom:33%;" />

By defining the loss and the labels, we can achieve the following goal:

<img src="assets/image-20240513134431998.png" alt="image-20240513134431998" style="zoom:33%;" />

This is almost like supervised learning. We can train the Actor by getting a dataset:

<img src="assets/image-20240513134708863.png" alt="image-20240513134708863" style="zoom:33%;" />

We can also redefine the loss by introducing weights for each {observation, action} pair. For example, we are more desired to see $\hat{a}_1$ followed by $s_1$ than $\hat{a}_3$ followed by $s_3$.

$$
L = \sum A_n e_n
$$

<img src="assets/image-20240513135117659.png" alt="image-20240513135117659" style="zoom: 33%;" />

The difficulty is what determines $A_i$ and what $\{s_i, \hat{a}_i\}$ pairs to generate.

### Version 0

<img src="assets/image-20240515154630524.png" alt="image-20240515154630524" style="zoom:28%;" />

We can start with a randomly-initialized Actor and then run it with *many episodes*. This will help collect us with some data. If we see a observation-action pair that produces positive reward, we will assign a positive $A_i$​ value to that pair.

However, generally speaking, this is not a very good strategy since actions are not independent. 

- An action affects the subsequent observations and thus subsequent rewards. 
- **Reward delay**: Actor has to sacrifice immediate reward to gain more long-term reward.
- In *space invader*, only “fire” yields positive reward, so vision 0 will learn an actor that always “fire”.

### Version 1

We need to take into account all the rewards that we gain after performing one action at timestamp $t$. Therefore, we can define a **cumulated reward** (as opposed to an immediate reward in the last version) $G_t$:

$$
A_t = G_t = \sum_{n=t}^N r_n
$$

<img src="assets/image-20240515155541817.png" alt="image-20240515155541817" style="zoom:33%;" />

### Version 2

However, if the sequence of length $N$ is very long, then $r_N$ if probably not the credit of $a_1$. Therefore, we can redefine $G_t'$ using a **discount factor** $\gamma < 1$:

**Discounted Cumulated Reward**:
$$
A_t = G_t' = \sum_{n=t}^N \gamma^{n-t} r_n
$$

### Version 3

But good or bad rewards are "relative." If all the $r_n \geq 10$, then having a reward of $10$ is actually very bad. We can redefine $A_i$ values by minusing by a **baseline** $b$. This makes $A_i$ to have positive and negative values.

$$
A_t = G_t' - b
$$

### Policy Gradient

1. Initialize Actor NN parameters $\theta^0$
2. For training iteration $i=1$ to $T$:
   - Using Actor $\theta^{i-1}$ to interact with the environment.
   - We then obtain data/training set $\{s_1, a_1\}, \{s_2, a_2\}, ..., \{s_N, a_N\}$​. Note that here the data collection is in the `for` loop of training iterations.
   - Compute $A_1, A_2, ..., A_N$
   - Compute Loss $L=\sum_n A_n e_n$
   - Update Actor parameters using gradient descent: $\theta^i \leftarrow \theta^{i-1} - \eta \nabla L$​

This process (from 2.1 to 2.3) is very expensive. Each time you update the model parameters, you need to collect the whole training set again. However, this process is necessary because it is based on the *experience* of Actor $\theta^{i-1}$。 We need to use a the new Actor $\theta^i$​'s *trajectory* to train the new Actor.

**On-Policy** Learning: <u>the actor to train</u> and <u>the actor for interacting</u> are the same.

**Off-Policy** Learning: <u>the actor to train</u> and <u>the actor for interacting</u> are different. In this way, we don't have to collect data after each update

- One example is **Proximal Policy Optimization (PPO)**. The actor to train has to know its difference with the actor to interact.

**Exploration**: The actor needs to have some *randomness* during data collection (remember that we sample our action). If the initial setting of the actor is always performing one action, then we  will never know whether other actions are good or bad. If we don't have randomness, we can't really train the actor. With exploration, we can collect more diverse data. We sometimes want to even deliberately add randomness. We can **enlarge output entropy** or **add noises onto parameters of the actor**.

### Critic

Critic: Given actor $\theta$, how good it is when observing $s$ (and taking action $a$​).

- An example is **value function** $V^{\theta}(s)$: When using actor $\theta$, the **discounted *cumulated* reward** (see Version 2) expected to be obtained after seeing $s$. Note that since the function depends on $\theta$, the same observation $s$ will have a different associated value if the actors are different. **The output values of a critic depend on the actor evaluated**.

<img src="assets/image-20240515215300373.png" alt="image-20240515215300373" style="zoom:33%;" />

### MC & TD

How to train a critic?

One way is to use **Monte-Carlo (MC)** based approach: The critic watches actor $\theta$ to interact with the environment. However, we can only update $V^{\theta}$ after the episode is over.

<img src="assets/image-20240515215625579.png" alt="image-20240515215625579" style="zoom:35%;" />

$V^{\theta}(s_a)$ should be as close as possible to $G'_a$ and $V^{\theta}(s_b)$ should be as close as possible to $G'_b$.

Another way is **Temporal-difference (TD)** approach. We don't need to wait for the entire episode.

<img src="assets/image-20240515220314195.png" alt="image-20240515220314195" style="zoom:35%;" />

MC and TD may produce *different* results for the same $V^{\theta}$​ with the same dataset as seen from the following example in *Sutton* v2 (Example 6.4). 

That is because MC and TD have different assumptions. For TD, we assume that $s_a$ can *not* influence $s_b$ (no correlation). It is possible that our dataset (the samples) is limited. For MC, we assume that $s_a$ can influence $s_b$. It is possible that observing $s_a$ influences the reward of observing $s_b$ so that it is always zero.

<img src="assets/image-20240516212052542.png" alt="image-20240516212052542" style="zoom:38%;" />

### Version 3.5

How can we use critic on training actor?

Recall that in Version 3, we introduce a baseline $b$ as normalization. How to choose $b$? One possible way is to set $b = V^{\theta}(s_t)$.
$$
A_t = G_t' - V^{\theta}(s_t)
$$
<img src="assets/image-20240516213342601.png" alt="image-20240516213342601" style="zoom:35%;" />

Why this method works? Remember that $V^{\theta}(s_t)$ takes into account many episodes, so it's an expected value. This makes $A_t$ very intuitive.

<img src="assets/image-20240516213852293.png" alt="image-20240516213852293" style="zoom:38%;" />

### Version 4: Advantage Actor-Critic

However, Version 3 is still problematic because $G_t'$ may be very random (it's just a sample). It may be very good or bad. We want it to resemble an average value like $V^{\theta}(s_t)$​​​.

Therefore, for a pair $\{ s_t, a_t \}$:
$$
A_t = \left[r_t + V^{\theta}(s_{t+1})\right] - V^{\theta}(s_t)
$$
Note that $s_{t+1}$ is the observation of the environment influenced by the actor taking action $a_t$.

<img src="assets/image-20240516214628159.png" alt="image-20240516214628159" style="zoom:36%;" />

The parameters of actor and critic can be shared. In practice, the first several layers are shared in common. If the observation is an image, then the convolutional layers will be shared.

<img src="assets/image-20240516220033541.png" alt="image-20240516220033541" style="zoom:33%;" />

### Reward Shaping

In tasks like robot arm bolting on the screws, the reward $r_t$ is always or mostly $0$​ (**spare reward**) because the actor is randomly initialized and it will very likely fail to complete the task. So, we may not know what actions are good or bad. As a consequence, developers can define *extra* rewards to guide agents. This is called **reward shaping**.

For example, we can incentive the actor to do different things by defining extra rewards based on our *domain knowledge*.

<img src="assets/image-20240517182146078.png" alt="image-20240517182146078" style="zoom:33%;" />

We can also add **curiosity** as an extra reward: obtaining extra reward when the agent sees something new (but meaningful).

### Imitation Learning

Defining a reward can be challenging in some tasks. Hand-crafted rewards can lead to uncontrolled behavior.

We can use Imitation Learning. We assume that actor can interact with the environment, but reward function is not available.

<img src="assets/image-20240517183746682.png" alt="image-20240517183746682" style="zoom:33%;" />

We can record demonstration of an expert (for example recording of human drivers in self-driving cars training). We define $\{\hat{\tau}_1, ..., \hat{\tau}_K\}$, where each $\hat{\tau}_i$​​ is one trajectory of the expert. Isn't this supervised learning? Yes, this is known as **Behavior Cloning**. This method has several problems:

- The experts only sample limited observation. The actor will not know what to do in other edge cases.
- The agent will copy every behavior, even irrelevant actions.

### Inverse RL (IRL)

This is vanilla RL:

<img src="assets/image-20240517184906946.png" alt="image-20240517184906946" style="zoom:25%;" />

Inverse RL has the exact *opposite* direction of process. It first learns the reward function and then uses that reward function to find the optimal actor.

<img src="assets/image-20240517185033513.png" alt="image-20240517185033513" style="zoom:33%;" />

The principle is that the teacher is always the best. As a result, the basic algorithm is:

1. Initialize an actor.
2. In each iteration:
   1. The actor interacts with the environments to obtain some trajectories. 
   2. Define a reward function, which makes **the trajectories of the teacher better than the actor**. 
   3. The actor learns to maximize the reward based on the new reward function.
   4. Repeat.

3. Output the reward function and the actor learned from the reward function.

<img src="assets/image-20240517185753070.png" alt="image-20240517185753070" style="zoom:33%;" />

IRL is very similar to GAN:

<img src="assets/image-20240517185920053.png" alt="image-20240517185920053" style="zoom:33%;" />

# 5/27 Lecture 13: Network Compression

## Network Pruning and Lottery Ticket Hypothesis

We may need to deploy ML models in resource-constrained environments because of lower-latency and privacy concerns.

NN are typically *over-parameterized* -- there's significant redundant weights or neurons.

How to evaluate the importance of a *weight*? We can either look at its **absolute value** or its $b_i$​ value as seen in life-long learning.

How to evaluate the importance of a *neuron*? We can record the number of times it didn't output a $0$​ on a given dataset.

After pruning, the accuracy will drop (hopefully not too much). We can then fine-tune on training data for recover. Remember not to prune too much at once, or the NN won’t recover. We can do the whole process iteratively.

<img src="assets/image-20240526130630137.png" alt="image-20240526130630137" style="zoom:33%;" />

**Weight pruning** is *not* always a very effective method. The network architecture becomes irregular (not fully-connected anymore). It also becomes harder to implement and harder to speed up (if you set the pruned weights to $0$, then you are not actually making the NN any smaller).

<img src="assets/image-20240526131333421.png" alt="image-20240526131333421" style="zoom:30%;" />

**Neuron pruning** is a *better* method as it helps the NN architecture stay regular. It's also easy to implement and easy to speedup.

<img src="assets/image-20240526131735594.png" alt="image-20240526131735594" style="zoom:30%;" />

How about simply train a smaller network? 

It is widely known that smaller network is more difficult to learn successfully. It is easier to first train a large NN and then gradually reduce its size.

<img src="assets/image-20240526134830493.png" alt="image-20240526134830493" style="zoom:30%;" />

<img src="assets/image-20240526135040715.png" alt="image-20240526135040715" style="zoom:33%;" />

The **Lottery Ticket Hypothesis** states that a large network is like a bunch of sub-networks so that we have a higher chance of train the network successfully. 

## Other ways

### Knowledge Distillation

There are two NNs: one large Teacher Net, one small Student Net.

<img src="assets/image-20240526142007275.png" alt="image-20240526142007275" style="zoom:33%;" />

The Student Net can hopefully learn the information that "1" is similar to "7" from the output of the Teacher Net. This is better than simply learning from the dataset labels. In practice, we may also use an ensemble NN to represent the Teacher Net. 

We may add **Temperature for softmax** $T$, a *hyperparameter*, when doing Knowledge Distillation:

$$
y_i' = \frac{\exp{\frac{y_i}{T}}}{\sum_j \exp{\frac{y_j}{T}}}
$$

<img src="assets/image-20240526143053309.png" alt="image-20240526143053309" style="zoom:33%;" />

The temperature can help make sharp distribution a little bit *smoother*. This is critical because vanilla softmax can produce results very similar to the one-hot label in the dataset.

### Parameter Quantization

#### Less Bits

We can use less bits to represent a value (lowering the numerical precision).

#### Weight Clustering

<img src="assets/image-20240526145421852.png" alt="image-20240526145421852" style="zoom:38%;" />

#### Huffman encoding

We can represent frequent clusters by less bits, represent rare clusters by more bits.

#### Binary Weights

It's even possible to make your weights always $+1$ or $-1$​. One classical example is Binary Connect.

### Architecture Design: Depthwise Seperable Convolution

#### Depthwise Convolution

Depthwise convolution considers the interaction within a specific channel.

<img src="assets/image-20240526150741632.png" alt="image-20240526150741632" style="zoom:33%;" />

The filter number must be equal to the input channel number (each filter only considers one channel). The number of input channels must be equal to the number of output channels. The filters are $k \times k$ matrices. 

However, one problem is that there is no interaction between channels.

#### Pointwise Convolution

Pointwise convolution considers the interaction amongst channels.

<img src="assets/image-20240526151334552.png" alt="image-20240526151334552" style="zoom:36%;" />

The filter number does not have to be equal to the input channel number. However, we force the filter size to be $1 \times 1$.

#### Number of Parameters Comparison

<img src="assets/image-20240526151706549.png" alt="image-20240526151706549" style="zoom:38%;" />

Since $O$ is usually a very large number, we can safely ignore $\frac{1}{O}$. Therefore, the standard CNN uses $k^2$ times more parameters than Depthwise Seperable Convolution.

#### Low rank approximation

$$
M = WN=U(VN)=(UV)N
$$

Therefore, we can approximate the matrix $W=UV$ though there're some constraints on $W$ (it's low-rank).

<img src="assets/image-20240526152119201.png" alt="image-20240526152119201" style="zoom:38%;" />

The Depthwise Seperable Convolution is very similar to this idea of using two layers to replace one layer in order to reduce the parameter size.

<img src="assets/image-20240526153043698.png" alt="image-20240526153043698" style="zoom:33%;" />

