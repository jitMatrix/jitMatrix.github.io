---
title: 'R for Deep Learning (IV): Build Convolutional Neural Network from Scratch'
url: 1402.html
id: 1402
categories:
  - Uncategorized
tags:
---

\[mathjax\] Here, we will briefly introduce convolutional neural-network (CNN). CNNs were first made popular in 1998 by LeCun's seminal paper. Since then, they have proven to be the best method we have for recognising patterns in images, sounds, videos, and even text! Image recognition was initially a manual process; researchers would have to specify which bits (features) of an image were useful to identify. For example, if we wanted to classify an image into ‘cat’ or ‘basketball’ we could have created code that extracts colours (basketballs are orange) and shapes (cats have triangular ears). Perhaps with a count of these features we could then run a linear regression to get the relationship between number of triangles and whether the image is a cat or a tree. This approach suffers from issues of image scale, angle, quality and light. Scale Invariant Feature Transformation (SIFT) largely improved upon this and was used to provide a `feature description' of an object, which could then be fed into a linear regression (or any other relationship learner). However, this approach had set-in-stone rules that could not be optimally altered for a specific domain. CNNs look at images (extract features) in an interesting way. To start, they look only at very small parts of an image (at a time), perhaps through a restricted window of 5 by 5 pixels (a filter). 2D convolutions are used for images, and these slide the window across until the whole image has been covered. This stage would typically extract colours and edges. However, the next layer of the network would look at a combination of the previous filters and thus 'zoom-out'. After a certain number of layers the network would be 'zoomed-out' enough to recognise shapes and larger structures. In this post, we will implement convolutional (CONV) and pooling (POOL) layers in R, including both forward propagation and backward propagation.  

### **Notation:**

\- Superscript $\[l\]$ denotes an object of the $l^{th}$ layer. **  Example:** $a^{\[4\]}$ is the $4^{th}$ layer activation. $W^{\[5\]}$ and $b^{\[5\]}$ are the $5^{th}$ layer parameters. - Superscript $(i)$ denotes an object from the $i^{th}$ example. **  Example: **$x^{(i)}$ is the $i^{th}$ training example input. - Lowerscript $i$ denotes the $i^{th}$ entry of a vector. **  Example: **$a^{\[l\]}\_i$ denotes the $i^{th}$ entry of the activations in layer $l$, assuming this is a fully connected (FC) layer. - $n\_H$, $n\_W$ and $n\_C$ denote respectively the height, width and number of channels of a given layer. If we want to reference a specific layer $l$, we can also write $n\_H^{\[l\]}$, $n\_W^{\[l\]}$, $n\_C^{\[l\]}$. - $n\_{H_{prev}}$, $n_{W_{prev}}$ and $n_{C_{prev}}$ denote respectively the height, width and number of channels of the previous layer. If referencing a specific layer $l$, this could also be denoted $n\_H^{\[l-1\]}$, $n\_W^{\[l-1\]}$, $n_C^{\[l-1\]}$.   We assume that we are already familiar with **\`R\`** and/or have read the [previous post of DNN](http://www.parallelr.com/r-deep-neural-network-from-scratch/). Let's get started!  

1 - Outline of this Post
========================

 

We will be implementing the building blocks of a convolutional neural network! Each function we will implement will have detailed instructions that will walk we through the steps needed:

\- Convolution functions, including:

      \- Padding with Zero

      \- Convolve window

      \- Convolution forward

      \- Convolution backward

\- Activation functions, including:

      \- Sigmoid function

      \- Relu function

\- Pooling functions, including:

      \- Pooling forward

      \- Create mask

      \- Distribute value

      \- Pooling backward

\- Affine function, including:

      \- Affine forward

      \- Affine backward

\- Cost function:

      \- Softmax function

      \- Cross-entropy loss function

  This notebook will ask we to implement these functions from scratch in **\`R\`**. Then, we will use these functions to build the following model:

 

![](http://www.parallelr.com/wp-content/uploads/2017/12/model-1024x348.png)

**Note** that for every forward function, there is its corresponding backward equivalent. Hence, at every step of wer forward module we will store some parameters in a cache. These parameters are used to compute gradients during backpropagation.  

2 - Convolutional Neural Networks
=================================

  Although programming frameworks make convolutions easy to use, they remain one of the hardest concepts to understand in Deep Learning. A convolution layer transforms an input volume into an output volume of different size, as shown below.

![](http://www.parallelr.com/wp-content/uploads/2017/12/conv_nn.png)

In this post, we will build every step of the convolution layer. we will first implement two helper functions: one for zero padding and the other for computing the convolution function itself.  

2.1 - Padding with Zero
-----------------------

  Padding with Zero adds zeros around the border of an image:

![](http://www.parallelr.com/wp-content/uploads/2017/12/PAD-1024x844.png)

\*\*Figure 1\*\* : \*\*Padding with Zero\*\* Image (3 channels, RGB) with a padding of 2.

The main benefits of padding are the following: - It allows we to use a CONV layer without necessarily shrinking the height and width of the volumes. This is important for building deeper networks, since otherwise the height/width would shrink as we go to deeper layers. An important special case is the "same" convolution, in which the height/width is exactly preserved after one layer. - It helps us keep more of the information at the border of an image. Without padding, very few values at the next layer would be affected by pixels as the edges of an image.

\[code language="R"\] # FUNCTION: padding for 3-D array pad3d <- function(input=NULL, stride=1) { input\_N <- dim(input)\[1\] input\_H <- dim(input)\[2\] input\_W <- dim(input)\[3\] input\_C <- dim(input)\[4\] output <- array(0,c(input\_N,input\_H+2\*stride,input\_W+2\*stride,input\_C)) for (i in 1:input\_N) { for (j in 1:input\_C) { output\[i,,,j\] <- rbind(matrix(0,stride,input\_W+2\*stride), cbind(matrix(0,input\_H,stride),input\[i,,,j\],matrix(0,input\_H,stride)), matrix(0,stride,input\_W+2\*stride)) } } return(output) } \[/code\]  

2.2 - Single step of convolution
--------------------------------

  In this part, implement a single step of convolution, in which we apply the filter to a single position of the input. This will be used to build a convolutional unit, which: - Takes an input volume - Applies a filter at every position of the input - Outputs another volume (usually of different size)

![](http://www.parallelr.com/wp-content/uploads/2017/12/Convolution_schematic.gif)

\*\*Figure 2\*\* : \*\*Convolution operation\*\* with a filter of 2x2 and a stride of 1 (stride = amount we move the window each time we slide)

In a computer vision application, each value in the matrix on the left corresponds to a single pixel value, and we convolve a 3x3 filter with the image by multiplying its values element-wise with the original matrix, then summing them up. In this first step of the exercise, we will implement a single step of convolution, corresponding to applying a filter to just one of the positions to get a single real-valued output.Later in this notebook, we'll apply this function to multiple positions of the input to implement the full convolutional operation.   \[code language="R"\] # FUNCTION: conv\_single\_step conv\_single\_step <- function(a\_slice\_prev, W, b){ # Apply one filter defined by parameters W on a single slice (a\_slice\_prev) of the output activation # of the previous layer. # # Arguments: # a\_slice\_prev -- slice of input data of shape (f, f, n\_C\_prev) # W -- Weight parameters contained in a window - matrix of shape (f, f, n\_C\_prev) # b -- Bias parameters contained in a window - matrix of shape (1, 1, 1) # # Returns: # Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data f <- dim(a\_slice\_prev)\[1\] n\_C\_prev <- dim(a\_slice\_prev)\[3\] W <- array(W,c(f, f, n\_C\_prev)) s <- a\_slice\_prev * W + b Z <- sum(s) return(Z) } \[/code\]  

2.3 - Convolutional Neural Networks - Forward pass
--------------------------------------------------

 

In the forward pass, we will take many filters and convolve them on the input. Each 'convolution' gives we a 2D matrix output. we will then stack these outputs to get a 3D volume:

 

\[video width="2560" height="1600" mp4="http://www.parallelr.com/wp-content/uploads/2017/12/conv_kiank.mp4" loop="true"\]\[/video\]

  The function below is designed to to convolve the filters W on an input activation A\_prev. This function takes as input A\_prev, the activations output by the previous layer (for a batch of m inputs), F filters/weights denoted by W, and a bias vector denoted by b, where each filter has its own (single) bias. Finally we also have access to the hyperparameters dictionary which contains the stride and the padding. **Hint:** 1\. To select a slice at the upper left corner of a matrix "a\_prev" This will be useful when we will define \`a\_slice\_prev\` below, using the \`start/end\` indexes we will define. 2. To define a\_slice we will need to first define its corners \`vert\_start\`, \`vert\_end\`, \`horiz\_start\` and \`horiz\_end\`. This figure may be helpful for we to find how each of the corner can be defined using h, w, f and s in the code below.

![](http://www.parallelr.com/wp-content/uploads/2017/12/vert_horiz_kiank.png)

\*\*Figure 3\*\* : \*\*Definition of a slice using vertical and horizontal start/end (with a 2x2 filter)\*\* This figure shows only a single channel.

**Reminder** The formulas relating the output shape of the convolution to the input shape is:

$$ n\_H = \\lfloor \\frac{n\_{H_{prev}} - f + 2 \\times pad}{stride} \\rfloor +1 $$

$$ n\_W = \\lfloor \\frac{n\_{W_{prev}} - f + 2 \\times pad}{stride} \\rfloor +1 $$

$$ n_C = \\text{number of filters used in the convolution}$$

For this post, we won't worry about vectorization, and will just implement everything with for-loops. \[code language="R"\] # FUNCTION: conv\_forward conv\_forward <- function(A\_prev, W, b, hparameters){ # Implements the forward propagation for a convolution function # # Arguments: # A\_prev -- output activations of the previous layer, array of shape (m, n\_H\_prev, n\_W\_prev, n\_C\_prev) # W -- Weights, array of shape (f, f, n\_C\_prev, n\_C) # b -- Biases, array of shape (1, 1, 1, n\_C) # hparameters -- R list containing "stride" and "pad" # # Returns: # Z -- conv output, array of shape (m, n\_H, n\_W, n\_C) # cache -- cache of values needed for the conv\_backward() function # Retrieve dimensions from A\_prev's shape m <- dim(A\_prev)\[1\] n\_H\_prev <- dim(A\_prev)\[2\] n\_W\_prev <- dim(A\_prev)\[3\] n\_C\_prev <- dim(A\_prev)\[4\] # Retrieve dimensions from W's shape f <- dim(W)\[1\] n\_C\_prev <- dim(W)\[3\] n\_C <- dim(W)\[4\] # Retrieve information from "hparameters" stride <- hparameters$stride pad <- hparameters$pad # Compute the dimensions of the CONV output volume using the formula given above. n\_H = floor((n\_H\_prev-f+2\*pad)/stride) + 1 n\_W = floor((n\_W\_prev-f+2\*pad)/stride) + 1 # Initialize the output volume Z with zeros. Z <- array(0,c(m, n\_H, n\_W, n\_C)) # Create A\_prev\_pad by padding A\_prev A\_prev\_pad <- pad3d(A\_prev, pad) for (i in 1:m) { # loop over the batch of training examples a\_prev\_pad <- A\_prev\_pad\[i,,,\] # Select ith training example's padded activation if (is.matrix(a\_prev\_pad)==TRUE) { a\_prev\_pad <- array(a\_prev\_pad, c(dim(a\_prev\_pad)\[1\],dim(a\_prev\_pad)\[2\],1)) } for (h in 1:n\_H) { # loop over vertical axis of the output volume for (w in 1:n\_W) { # loop over horizontal axis of the output volume for (c in 1:n\_C) { # loop over channels (= #filters) of the output volume # Find the corners of the current "slice" vert\_start <- (h - 1) * stride + 1 vert\_end <- vert\_start + f - 1 horiz\_start <- (w - 1) * stride + 1 horiz\_end <- horiz\_start + f - 1 # Use the corners to define the (3D) slice of a\_prev\_pad a\_slice\_prev <- a\_prev\_pad\[vert\_start:vert\_end,horiz\_start:horiz\_end,\] if (is.matrix(a\_slice\_prev)==TRUE) { a\_slice\_prev <- array(a\_slice\_prev, c(dim(a\_slice\_prev)\[1\],dim(a\_slice\_prev)\[2\],1)) } # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. Z\[i, h, w, c\] = conv\_single\_step(a\_slice\_prev, W\[,,,c\], b\[,,,c\]) } } } } # Save information in "cache" for the backprop cache <- list(A\_prev=A\_prev, W=W, b=b, hparameters=hparameters) list(Z=Z,cache=cache) } \[/code\]  

3 - Activation
==============

  CONV layer should also contain an activation, in which case we would add the following line of code: \[code language="R"\] # Convolve the window to get back one output neuron Z = ... # Apply activation A = activation(Z) \[/code\] We will use two activation functions: **\- Sigmoid:** $\\sigma(Z) = \\sigma(W A + b) = \\frac{1}{ 1 + e^{-(W A + b)}}$. We have provided you with the \`sigmoid\` function. This function returns **two** items: the activation value "\`a\`" and a "\`cache\`" that contains "\`Z\`". **\- ReLU:** The mathematical formula for ReLu is $A = RELU(Z) = max(0, Z)$. We have provided you with the \`relu\` function. This function returns **two **items: the activation value "\`A\`" and a "\`cache\`" that contains "\`Z\`" . Below are sigmoid and relu activation function. \[code language="R"\] sigmoid <- function(Z){ # Implements the sigmoid activation in R # # Arguments: # Z -- array of any shape # # Returns: # A -- output of sigmoid(z), same shape as Z # cache -- returns Z as well, useful during backpropagation A <- 1/(1+exp(-Z)) cache <- Z list(A = A, cache = cache) } sigmoid\_backward <- function(dA, cache){ # Implement the backward propagation for a single SIGMOID unit. # # Arguments: # dA -- post-activation gradient, of any shape # cache -- 'Z' where we store for computing backward propagation efficiently # # Returns: # dZ -- Gradient of the cost with respect to Z Z <- cache s <- 1/(1 + exp(-Z)) dZ <- dA * s * (1-s) return(dZ) } relu <- function(Z){ # Implement the RELU function. # # Arguments: # Z -- Output of the linear layer, of any shape # # Returns: # A -- Post-activation parameter, of the same shape as Z # cache -- a R list containing "A" ; stored for computing the backward pass efficiently A <- pmax(Z, 0) cache <- Z list(A = A, cache = cache) } relu\_backward <- function(dA, cache){ # Implement the backward propagation for a single RELU unit. # # Arguments: # dA -- post-activation gradient, of any shape # cache -- 'Z' where we store for computing backward propagation efficiently # # Returns: # dZ -- Gradient of the cost with respect to Z Z <- cache dZ <- array(dA, dim = dim(dA)) dZ\[Z < 0\] <- 0 return(dZ) } \[/code\]  

4 - Pooling layer
=================

  The pooling (POOL) layer reduces the height and width of the input. It helps reduce computation, as well as helps make feature detectors more invariant to its position in the input. The two types of pooling layers are: - Max-pooling layer: slides an ($f, f$) window over the input and stores the max value of the window in the output. - Average-pooling layer: slides an ($f, f$) window over the input and stores the average value of the window in the output.  

![](http://www.parallelr.com/wp-content/uploads/2017/12/max_pool1-1-1024x704.png) ![](http://www.parallelr.com/wp-content/uploads/2017/12/a_pool-1-1024x685.png)

\*\*Figure 4\*\* : \*\*Pooling Operation\*\* Max pooling and average pooling.

These pooling layers have no parameters for backpropagation to train. However, they have hyperparameters such as the window size $f$. This specifies the height and width of the fxf window you would compute a max or average over. Now, you are going to implement MAX-POOL and AVG-POOL, in the same function. As there's no padding, the formulas binding the output shape of the pooling to the input shape is:

$$ n\_H = \\lfloor \\frac{n\_{H_{prev}} - f}{stride} \\rfloor +1 $$

$$ n\_W = \\lfloor \\frac{n\_{W_{prev}} - f}{stride} \\rfloor +1 $$

$$ n\_C = n\_{C_{prev}}$$

  \[code language="R"\] pool\_forward <- function(A\_prev, hparameters, mode = "max"){ # Implements the forward pass of the pooling layer # # Arguments: # A\_prev -- Input data, array of shape (m, n\_H\_prev, n\_W\_prev, n\_C\_prev) # hparameters -- R list containing "f" and "stride" # mode -- the pooling mode you would like to use, defined as a string ("max" or "average") # # Returns: # A -- output of the pool layer, a numpy array of shape (m, n\_H, n\_W, n\_C) # cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters # Retrieve dimensions from A\_prev's shape m <- dim(A\_prev)\[1\] n\_H\_prev <- dim(A\_prev)\[2\] n\_W\_prev <- dim(A\_prev)\[3\] n\_C\_prev <- dim(A\_prev)\[4\] # Retrieve information from "hparameters" stride <- hparameters$stride f <- hparameters$f # Define the dimensions of the output n\_H = floor(1 + (n\_H\_prev - f) / stride) n\_W = floor(1 + (n\_W\_prev - f) / stride) n\_C = n\_C\_prev A = array(0, c(m, n\_H, n\_W, n\_C)) for (i in 1:m) { # loop over the batch of training examples for (h in 1:n\_H) { # loop over vertical axis of the output volume for (w in 1:n\_W) { # loop over horizontal axis of the output volume for (c in 1:n\_C) { # loop over channels of the output volume # Find the corners of the current "slice" vert\_start <- (h - 1) * stride + 1 vert\_end <- vert\_start + f - 1 horiz\_start <- (w - 1) * stride + 1 horiz\_end <- horiz\_start + f - 1 # Use the corners to define the current slice on the ith training example of A\_prev, channel c. a\_prev\_slice = A\_prev\[i,vert\_start:vert\_end,horiz\_start:horiz\_end,c\] if (is.matrix(a\_prev\_slice)==TRUE) { a\_prev\_slice <- array(a\_prev\_slice, c(dim(a\_prev\_slice)\[1\],dim(a\_prev\_slice)\[2\],1)) } # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. A\[i, h, w, c\] <- ifelse(mode == "max", max(a\_prev\_slice), mean(a\_prev\_slice)) } } } } # Store the input and hparameters in "cache" for pool\_backward() cache = list(A\_prev=A_prev, hparameters=hparameters) list(A=A, cache=cache) } \[/code\]  

5 - Affine layer module
=======================

  Before implementing the affine layer module, we should first stretch the data to vectors. \[code language="R"\] # array to col arr2col <- function(X){ output <- array(X,c(dim(X)\[1\],(dim(X)\[2\]\*dim(X)\[3\]\*dim(X)\[4\]))) cache <- X list(output = output, cache = cache) } # col to array col2arr <- function(X, cache){ output <- array(X,dim = dim(cache)) } \[/code\]   Assume that you have initialized your parameters(such as [He initialization](https://arxiv.org/abs/1502.01852)), you will do the affine layer module. You will start by implementing some basic functions that you will use later when implementing the model. You will complete three functions in this order: **\- LINEAR** **\- LINEAR -> ACTIVATION where ACTIVATION will be either ReLU or Sigmoid.** The Affine forward module (vectorized over all the examples) computes the following equations: $$Z^{\[l\]} = W^{\[l\]}A^{\[l-1\]} +b^{\[l\]}\\tag{4}$$ where $A^{\[0\]} = X$. **Reminder:** The mathematical representation of this unit is $Z^{\[l\]} = W^{\[l\]}A^{\[l-1\]} +b^{\[l\]}$.

\[code language="R"\] affine\_forward <- function(A\_prev, W, b){ # Implement the linear part of a layer's forward propagation. # # Arguments: # A -- activations from previous layer (or input data): (number of examples, size of previous layer) # W -- weights matrix: numpy array of shape (size of previous layer, size of current layer) # b -- bias vector, numpy array of shape (1, size of the current layer) # # Returns: # Z -- the input of the activation function, also called pre-activation parameter # cache -- a R list containing "A", "W" and "b" ; stored for computing the backward pass efficiently Z <- sweep(A\_prev %*% W, 2, b, '+') cache <- list(A\_prev = A\_prev, W = W, b = b) list(Z = Z, cache = cache) } \[/code\] The affine activation module will computes the following equations: $$A^{\[l\]} = activation(Z^{\[l\]})$$ where ACTIVATION will be either ReLU or Sigmoid. \[code language="R"\] affine\_activation\_forward <- function(A\_prev, W, b, activation = "relu"){ # Implement the forward propagation for the LINEAR->ACTIVATION layer # # Arguments: # A\_prev -- activations from previous layer (or input data): (size of previous layer, number of examples) # W -- weights matrix: numpy array of shape (size of current layer, size of previous layer) # b -- bias vector, numpy array of shape (size of the current layer, 1) # activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu" # # Returns: # A -- the output of the activation function, also called the post-activation value # cache -- a R list containing "linear\_cache" and "activation\_cache"; # stored for computing the backward pass efficiently if (activation == "sigmoid") { affine\_forward\_output <- affine\_forward(A\_prev, W, b) Z <- affine\_forward\_output$Z affine\_cache <- affine\_forward\_output$cache sigmoid\_output <- sigmoid(Z) A <- sigmoid\_output$A activation\_cache <- sigmoid\_output$cache } else if (activation == "relu") { affine\_forward\_output <- affine\_forward(A\_prev, W, b) Z <- affine\_forward\_output$Z affine\_cache <- affine\_forward\_output$cache relu\_output <- relu(Z) A <- relu\_output$A activation\_cache <- relu\_output$cache } else if(activation == "NULL"){ affine\_forward\_output <- affine\_forward(A\_prev, W, b) A <- affine\_forward\_output$Z activation\_cache <- affine\_cache <- affine\_forward\_output$cache } cache = list(affine\_cache = affine\_cache, activation\_cache = activation_cache) list(A = A, cache = cache) } \[/code\]  

6 - Cost function
=================

  Now you will implement forward and backward propagation. You need to compute the cost, because you want to check if your model is actually learning.  

6.1 - Softmax function
----------------------

  The softmax function, or normalized exponential function, is a generalization of the logistic function that "squashes" a K-dimensional vector $z$, of arbitrary real values to a K-dimensional vector $\\sigma(Z)$ of real values in the range \[0, 1\] that add up to 1. The function is given by $$ \\sigma(Z)_{j} = \\frac{e^{Z_{j}}}{\\sum_{k=1}^{K}e^{z_{k}}}\\qquad for\ j = 1, …, K. $$ \[code language="R"\] softmax <- function(X){ score.exp <- exp(X) probs <- score.exp/rowSums(score.exp) cache <- X list(probs = probs, cache = cache) } \[/code\]  

6.2 - Cross-entropy loss
------------------------

  In binary classification, where the number of classes MM equals 2, cross-entropy can be calculated as: $$-{(y\\log(p) + (1 - y)\\log(1 - p))}$$ If $M > 2$ (i.e. multiclass classification), we calculate a separate loss for each class label per observation and sum the result. $$-\\sum_{c=1}^My_{o,c}\\log(p_{o,c})$$ **Note:** \- M - number of classes (dog, cat, fish) - log - the natural log - y - binary indicator (0 or 1) if class label $c$ is the correct classification for observation $o$ - p - predicted probability observation $o$ is of class $c$ \[code language="R"\] cross\_entropy\_cost <- function(AL, Y, batch\_size){ # create index for both row and col Y.set <- sort(unique(Y)) Y.index <- cbind(1:batch\_size, match(Y, Y.set)) corect.logprobs <- -log(AL\[Y.index\]) loss <- sum(corect.logprobs)/batch_size cache <- list(AL = AL, Y.index = Y.index) list(loss = loss, cache = cache) } \[/code\]  

7 - Backpropagation in convolutional neural networks
====================================================

  In modern deep learning frameworks, you only have to implement the forward pass, and the framework takes care of the backward pass, so most deep learning engineers don't need to bother with the details of the backward pass. The backward pass for convolutional networks is complicated. If you wish however, you can work through this portion of the post to get a sense of what backprop in a convolutional network looks like.  

7-1 Affine backward
-------------------

  Just as described in previous post, affine backward which means linear backward are based on chain rule in mathematic. \[code language="R"\] affine\_backward <- function(dZ, cache){ # Implement the linear portion of backward propagation for a single layer (layer l) # # Arguments: # dZ -- Gradient of the cost with respect to the linear output (of current layer l) # cache -- tuple of values (A\_prev, W, b) coming from the forward propagation in the current layer # # Returns: # dA\_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A\_prev # dW -- Gradient of the cost with respect to W (current layer l), same shape as W # db -- Gradient of the cost with respect to b (current layer l), same shape as b A\_prev <- cache$A\_prev W <- cache$W b <- cache$b m <- dim(A\_prev)\[1\] dW <- (t(A\_prev) %*% dZ) / m db <- (matrix(colSums(dZ),1,dim(b)\[2\])) / m dA\_prev <- dZ %*% t(W) list(dA\_prev = dA\_prev, dW = dW, db = db) } affine\_activation\_backward <- function(dA, cache, activation){ # Implement the backward propagation for the LINEAR->ACTIVATION layer. # # Arguments: # dA -- post-activation gradient for current layer l # cache -- tuple of values (linear\_cache, activation\_cache) we store for computing backward propagation efficiently # activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu" # # Returns: # dA\_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A\_prev # dW -- Gradient of the cost with respect to W (current layer l), same shape as W # db -- Gradient of the cost with respect to b (current layer l), same shape as b affine\_cache <- cache$affine\_cache activation\_cache <- cache$activation\_cache if (activation == "relu") { dZ <- relu\_backward(dA, activation\_cache) } else if (activation == "sigmoid") { dZ <- sigmoid\_backward(dA, activation\_cache) } else if (activation == "NULL") { dZ <- dA } affine\_backward\_output <- affine\_backward(dZ, affine\_cache) dA\_prev <- affine\_backward\_output$dA\_prev dW <- affine\_backward\_output$dW db <- affine\_backward\_output$db list(dA\_prev = dA_prev, dW = dW, db =db) } \[/code\]  

7.2 Pooling layer - backward pass
---------------------------------

  Next, let's implement the backward pass for the pooling layer, starting with the MAX-POOL layer. Even though a pooling layer has no parameters for backprop to update, you still need to backpropagation the gradient through the pooling layer in order to compute gradients for layers that came before the pooling layer.  

### 7.2.1 Max pooling - backward pass

  Before jumping into the backpropagation of the pooling layer, you are going to build a helper function called \`create\_mask\_from\_window()\` which does the following: $$ X = \\begin{bmatrix} 1 && 3 \\\ 4 && 2 \\end{bmatrix} \\quad \\rightarrow \\quad M =\\begin{bmatrix} 0 && 0 \\\ 1 && 0 \\end{bmatrix}\\tag{4}$$ As you can see, this function creates a "mask" matrix which keeps track of where the maximum of the matrix is. True (1) indicates the position of the maximum in X, the other entries are False (0). You'll see later that the backward pass for average pooling will be similar to this but using a different mask. \[code language="R"\] create\_mask\_from\_window <- function(x){ # Creates a mask from an input matrix x, to identify the max entry of x. # # Arguments: # x -- Array of shape (f, f) # # Returns: # mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x. mask <- (x == max(x)) + 0 } \[/code\] Why do we keep track of the position of the max? It's because this is the input value that ultimately influenced the output, and therefore the cost. Backprop is computing gradients with respect to the cost, so anything that influences the ultimate cost should have a non-zero gradient. So, backprop will "propagate" the gradient back to this particular input value that had influenced the cost.  

### 7.2.2 - Average pooling - backward pass

  In max pooling, for each input window, all the "influence" on the output came from a single input value--the max. In average pooling, every element of the input window has equal influence on the output. So to implement backprop, you will now implement a helper function that reflects this. For example if we did average pooling in the forward pass using a 2x2 filter, then the mask you'll use for the backward pass will look like: $$ dZ = 1 \\quad \\rightarrow \\quad dZ =\\begin{bmatrix} 1/4 && 1/4 \\\ 1/4 && 1/4 \\end{bmatrix}\\tag{5}$$ This implies that each position in the $dZ$ matrix contributes equally to output because in the forward pass, we took an average. \[code language="R"\] distribute\_value <- function(dz, shape){ # Distributes the input value in the matrix of dimension shape # # Arguments: # dz -- input scalar # shape -- the shape (n\_H, n\_W) of the output matrix for which we want to distribute the value of dz # # Returns: # a -- Array of size (n\_H, n\_W) for which we distributed the value of dz # Retrieve dimensions from shape n\_H <- shape\[1\] n\_W <- shape\[2\] # Compute the value to distribute on the matrix average <- dz / (n\_H * n\_W) # Create a matrix where every entry is the "average" value a = matrix(average, n\_H, n_W) return(a) } \[/code\]  

### 7.2.3 Putting it together: Pooling backward

  You now have everything you need to compute backward propagation on a pooling layer. \[code language="R"\] pool\_backward <- function(dA, cache, mode = "max"){ # Implements the backward pass of the pooling layer # # Arguments: # dA -- gradient of cost with respect to the output of the pooling layer, same shape as A # cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters # mode -- the pooling mode you would like to use, defined as a string ("max" or "average") # # Returns: # dA\_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A\_prev # Retrieve information from cache A\_prev <- cache$A\_prev hparameters <- cache$hparameters # Retrieve hyperparameters from "hparameters" stride <- hparameters$stride f <- hparameters$f # Retrieve dimensions from A\_prev's shape and dA's shape m <- dim(A\_prev)\[1\] n\_H\_prev <- dim(A\_prev)\[2\] n\_W\_prev <- dim(A\_prev)\[3\] n\_C\_prev <- dim(A\_prev)\[4\] m <- dim(dA)\[1\] n\_H <- dim(dA)\[2\] n\_W <- dim(dA)\[3\] n\_C <- dim(dA)\[4\] # Initialize dA\_prev with zeros dA\_prev <- array(0, c(m, n\_H\_prev, n\_W\_prev, n\_C\_prev)) for (i in 1:m) { # select training example from A\_prev a\_prev = A\_prev\[i,,,\] for (h in 1:n\_H) { for (w in 1:n\_W) { for (c in 1:n\_C) { # Find the corners of the current "slice" vert\_start <- (h - 1) * stride + 1 vert\_end <- vert\_start + f - 1 horiz\_start <- (w - 1) * stride + 1 horiz\_end <- horiz\_start + f - 1 # Compute the backward propagation in both modes. if(mode == "max"){ # Use the corners and "c" to define the current slice from a\_prev a\_prev\_slice <- a\_prev\[vert\_start:vert\_end,horiz\_start:horiz\_end,c\] # Create the mask from a\_prev\_slice mask <- create\_mask\_from\_window(a\_prev\_slice) # Set dA\_prev to be dA\_prev + (the mask multiplied by the correct entry of dA) dA\_prev\[i, vert\_start: vert\_end, horiz\_start: horiz\_end, c\] <- dA\_prev\[i, vert\_start: vert\_end, horiz\_start: horiz\_end, c\] + mask * dA\[i,h,w,c\] } else if(mode == "average"){ # Get the value a from dA da = dA\[i,h,w,c\] # Define the shape of the filter as fxf shape = c(f,f) # Distribute it to get the correct slice of dA\_prev. i.e. Add the distributed value of da. dA\_prev\[i, vert\_start: vert\_end, horiz\_start: horiz\_end, c\] <- dA\_prev\[i, vert\_start: vert\_end, horiz\_start: horiz\_end, c\] + distribute\_value(da, shape) } } } } } return(dA_prev) } \[/code\]  

7.3 - Convolutional layer backward pass
---------------------------------------

  Let's start by implementing the backward pass for a CONV layer.  

### 7.3.1 - Computing dA

  This is the formula for computing $dA$ with respect to the cost for a certain filter $W\_c$ and a given training example: $$ dA += \\sum \_{h=0} ^{n\_H} \\sum\_{w=0} ^{n\_W} W\_c \\times dZ_{hw} \\tag{1}$$ Where $W\_c$ is a filter and $dZ\_{hw}$ is a scalar corresponding to the gradient of the cost with respect to the output of the conv layer Z at the hth row and wth column (corresponding to the dot product taken at the ith stride left and jth stride down). Note that at each time, we multiply the the same filter $W\_c$ by a different dZ when updating dA. We do so mainly because when computing the forward propagation, each filter is dotted and summed by a different a\_slice. Therefore when computing the backprop for dA, we are just adding the gradients of all the a_slices.  

### 7.3.2 - Computing dW

  This is the formula for computing $dW\_c$ ($dW\_c$ is the derivative of one filter) with respect to the loss: $$ dW\_c += \\sum \_{h=0} ^{n\_H} \\sum\_{w=0} ^ {n\_W} a\_{slice} \\times dZ_{hw} \\tag{2}$$ Where $a_{slice}$ corresponds to the slice which was used to generate the acitivation $Z_{ij}$. Hence, this ends up giving us the gradient for $W$ with respect to that slice. Since it is the same $W$, we will just add up all such gradients to get $dW$.  

### 7.3.3 - Computing db

  This is the formula for computing $db$ with respect to the cost for a certain filter $W\_c$: $$ db = \\sum\_h \\sum\_w dZ\_{hw} \\tag{3}$$ As you have previously seen in basic neural networks, db is computed by summing $dZ$. In this case, you are just summing over all the gradients of the conv output (Z) with respect to the cost. \[code language="R"\] conv\_backward <- function(dZ, cache){ # Implement the backward propagation for a convolution function # # Arguments: # dZ -- gradient of the cost with respect to the output of the conv layer (Z), array of shape (m, n\_H, n\_W, n\_C) # cache -- cache of values needed for the conv\_backward(), output of conv\_forward() # # Returns: # dA\_prev -- gradient of the cost with respect to the input of the conv layer (A\_prev), # array of shape (m, n\_H\_prev, n\_W\_prev, n\_C\_prev) # dW -- gradient of the cost with respect to the weights of the conv layer (W) # array of shape (f, f, n\_C\_prev, n\_C) # db -- gradient of the cost with respect to the biases of the conv layer (b) # array of shape (1, 1, 1, n\_C) # Retrieve information from "cache" A\_prev <- cache$A\_prev W <- cache$W b <- cache$b hparameters <- cache$hparameters # Retrieve dimensions from A\_prev's shape m <- dim(A\_prev)\[1\] n\_H\_prev <- dim(A\_prev)\[2\] n\_W\_prev <- dim(A\_prev)\[3\] n\_C\_prev <- dim(A\_prev)\[4\] # Retrieve dimensions from W's shape f <- dim(W)\[1\] n\_C\_prev <- dim(W)\[3\] n\_C <- dim(W)\[4\] # Retrieve information from "hparameters" stride <- hparameters$stride pad <- hparameters$pad # Retrieve dimensions from dZ's shape m <- dim(dZ)\[1\] n\_H <- dim(dZ)\[2\] n\_W <- dim(dZ)\[3\] n\_C <- dim(dZ)\[4\] # Initialize dA\_prev, dW, db with the correct shapes dA\_prev <- array(0, c(m, n\_H\_prev, n\_W\_prev, n\_C\_prev)) dW <- array(0, c(f, f, n\_C\_prev, n\_C)) db <- array(0, c(1, 1, 1, n\_C)) # Pad A\_prev and dA\_prev A\_prev\_pad <- pad3d(A\_prev, pad) dA\_prev\_pad <- pad3d(dA\_prev, pad) for (i in 1:m) { # loop over the training examples # select ith training example from A\_prev\_pad and dA\_prev\_pad a\_prev\_pad <- A\_prev\_pad\[i,,,\] if (is.matrix(a\_prev\_pad)==TRUE) { a\_prev\_pad <- array(a\_prev\_pad, c(dim(a\_prev\_pad)\[1\],dim(a\_prev\_pad)\[2\],1)) } da\_prev\_pad <- dA\_prev\_pad\[i,,,\] if (is.matrix(da\_prev\_pad)==TRUE) { da\_prev\_pad <- array(da\_prev\_pad, c(dim(da\_prev\_pad)\[1\],dim(da\_prev\_pad)\[2\],1)) } for (h in 1:n\_H) { # loop over vertical axis of the output volume for (w in 1:n\_W) { # loop over horizontal axis of the output volume for (c in 1:n\_C) { # loop over the channels of the output volume # Find the corners of the current "slice" vert\_start <- (h - 1) * stride + 1 vert\_end <- vert\_start + f - 1 horiz\_start <- (w - 1) * stride + 1 horiz\_end <- horiz\_start + f - 1 # Use the corners to define the slice from a\_prev\_pad a\_slice <- a\_prev\_pad\[vert\_start:vert\_end, horiz\_start:horiz\_end, \] # Update gradients for the window and the filter's parameters using the code formulas given above da\_prev\_pad\[vert\_start:vert\_end, horiz\_start:horiz\_end, \] <- da\_prev\_pad\[vert\_start:vert\_end, horiz\_start:horiz\_end, \] + W\[,,,c\] * dZ\[i, h, w, c\] dW\[,,,c\] <- dW\[,,,c\] + a\_slice * dZ\[i, h, w, c\] db\[,,,c\] <- db\[,,,c\] + dZ\[i, h, w, c\] } } } # Set the ith training example's dA\_prev to the unpaded da\_prev\_pad (Hint: use X\[pad:-pad, pad:-pad, :\]) dA\_prev\[i, , , \] <- da\_prev\_pad\[(pad+1):(pad+n\_H\_prev), (pad+1):(pad+n\_W\_prev), \] } list(dA\_prev = dA\_prev, dW = dW, db = db) } \[/code\] Now that we have complete every main function in convolution model, we can use them to create a MNIST classifier, try it out!