---
title: 'R and OpenMP:  CRAN Package Modernization'
tags:
  - CRAN
  - knn
  - openMP
  - R
  - rstats
url: 895.html
id: 895
categories:
  - Accelerators
  - MultiCores
  - Performance Optimizaiton
date: 2016-08-15 02:55:52
---

* * *

_Indeed, a major current trend in R is what might be called “R+X”, where X is some other language or library._ _--Norman Matloff, Parallel Computing for Data Science_

* * *

Introduction
============

**Nowadays,** there are more than 8000 packages in CRAN (till April 2016 from Revolution Analytics  [Blog](http://blog.revolutionanalytics.com/2016/04/a-segmented-model-of-cran-package-growth.html)) and even more, packages locate in GitHub and R-forge. Meanwhile, most of us heavily rely on these packages for our daily analytical work. With the increasing amount of data as well as the diversification of data sources (such as image, speech, and video), data scientists will be experience performance issues.  Though there already have parallel solutions for us (check out [HPCR list](https://cran.r-project.org/web/views/HighPerformanceComputing.html)), we will fall into the case soon or later that no parallelized package is available in CRAN or GitHub. Therefore, we have to consider to implement specified parallel algorithm by ourselves. In general, you can design and implement the parallel algorithm from scratch or parallelize the existing packages.  If you are working on writing the parallel code from scratch, I recommend Prof. Matloff’s new book_, [Parallel Computing for Data Science](https://www.amazon.com/Parallel-Computing-Data-Science-Examples/dp/1466587016)_, and Dr. OTT’s blog _[R and openMP: boosting compiled code on multi-core cpu-s](http://www.parallelr.com/r-and-openmp-boosting-compiled-code-on-multi-core-cpu-s/)._ In this post, I will introduce the basic workflow and skills for accelerating legacy R package by OpenMP, and this produce is called [code modernization](https://software.intel.com/en-us/articles/what-is-code-modernization). Sometimes, acceleration legacy code is more challenge than writing parallel algorithm by yourself because we have to work on original software architecture and implementation, and try to minimal changes as much as possible. In below cycle chart, I illustrate several basic steps of code modernization and I am going to introduce each of steps in the below chapters with [KNN {class}](https://cran.r-project.org/web/packages/class/class.pdf) function, written by Prof. Brian Ripley, for example.   [![](http://www.parallelr.com/wp-content/uploads/2016/05/cycle-2.png)](http://www.parallelr.com/wp-content/uploads/2016/05/cycle-2.png)  

Understand Algorithm and Code
=============================

First and most important step before speedup code is to understand the original algorithm and implementation as much as possible ranging from data structure, functionality to coding. Now, let us start with our example of _class_ package and the major function of K nearest neighbors algorithm ([KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)). The algorithm of KNN can be shown as below chart. There are three steps. First, compute the distance from each observation of train dataset to each of test dataset, and then select K nearest from distance results (the direct approach is to sort the results and select first K elements). Finally, most voted group will win, or if several groups have the same accounts, just **RANDOM** breaking (Notes: this random breaking will cause little trouble in parallel version, consider why, and I will give more details later). [![](http://www.parallelr.com/wp-content/uploads/2016/05/KNN-3.png)](http://www.parallelr.com/wp-content/uploads/2016/05/KNN-3.png)   Next, we’re going through the source code and match each step of KNN with their implementations. Checking the body of R Knn, you can see the real algorithm is locate in C level and R function is only a wrapper. \[code language="r"\] >library(class) >knn function (train, test, cl, k = 1, l = 0, prob = FALSE, use.all = TRUE) { # skip Z <- .C(VR_knn, as.integer(k), as.integer(l), as.integer(ntr),....)) # skip } \[/code\] So, switching to VR_knn function in your local repository (class.c) or from GitHub ([here](https://github.com/cran/class/blob/master/src/class.c)), totally there are about 100 lines. And the code is organized very clearly and we can quickly identify the 3 steps with my highlight comments. \[code language="C"\] VR\_knn(Sint \*kin, Sint \*lin, Sint \*pntr, Sint \*pnte, Sint \*p, double \*train, Sint \*class, double \*test, Sint \*res, double \*pr, Sint \*votes, Sint \*nc, Sint \*cv, Sint \*use\_all) { // skip ... // Peng: main loop for (npat = 0; npat < nte; npat++) { // Peng: Step.1 -- compute distance for (k = 0; k < \*p; k++) { tmp = test\[npat + k \* nte\] - train\[j + k * ntr\]; dist += tmp * tmp; } // skip ... // Peng: Step.2 -- select K nearest /* Use 'fuzz' since distance computed could depend on order of coordinates */ if (dist <= nndist\[kinit - 1\] * (1 + EPS)) for (k = 0; k <= kn; k++) if (dist < nndist\[k\]) { for (k1 = kn; k1 > k; k1--) { nndist\[k1\] = nndist\[k1 - 1\]; pos\[k1\] = pos\[k1 - 1\]; } // Peng: Step.3 -- voting and breaking for (j = 0; j <= \*nc; j++) votes\[j\] = 0; if (\*use_all) { for (j = 0; j <; kinit; j++) // skip ... } &lt;span class="pl-k"&gt;else&lt;/span&gt; { &lt;span class="pl-c"&gt;/* break ties at random */ &lt;/span&gt; // skip ... } // Peng: Final results res\[npat\] = index; pr\[npat\] = (double) mm / (kinit + extras); } // Peng: end of main loop RANDOUT; } \[/code\]

Identify Bottleneck
===================

Two kinds of applications for parallelization can be divided into computer and memory intensive. OpenMP is used to resolve compute-intensive applications with multiple threads while message passing interface ([MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)) can be applied reduce local memory requirements by distributing the application in different machines. Therefore, we need to determine whether the package we are dealing with is compute-intensive and where the most time-consuming parts are. To do this, we can do a simple complexity analysis from algorithm view. Still using KNN for example as our description, we see the computational complexity in Step 1 is O(M\*N\*Q) since we iterate through train (M lines) and test (N lines) dataset and compute the distance with K features. Step 2 is sorting and the worst computational complexity is O(M\*K) and Step 3 is only the linear complexity of O(K). Regarding with memory requirements, the most size is the two input dataset with O(M\*Q) and O(N*Q) in Step 1. Totally, the computational complexity of KNN is **THREE** power while the memory usage is only **TWO** power. So, the OpenMP is appropriate.

Specify Parallel Region
=======================