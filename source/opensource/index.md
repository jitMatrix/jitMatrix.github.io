---
title: Open Sources
url: 6.html
id: 6
comments: false
date: 2016-01-21 05:52:36
---

 

#### [Parallel Grep](https://github.com/PatricZhao/ParallelGrep)

The PARALLEL grep is an enhancement for the standard grep under the Linux system. In this version, the multi-threads technology is used. The GPGPU will be applied in next version.  

#### [Parallel DNN](https://github.com/patriczhao/ParallelR)

Deep Neural Network (DNN) becomes a very popular approach in statistical analysis areas. Though there are several DNN packages in R, there almost can't use in practice for big data and deep neural network because the single core performance of R is really limited and the current design of DNN packages in R is not GPU-friendly. Our package will leverage GPU's power to accelerate DNN and we can get 10-50X speedup!  

#### RcuFFT

The NVIDIA CUDA Fast Fourier Transform library (cuFFT) provides a simple interface for computing FFTs up to 10x faster. By using hundreds of processor cores inside NVIDIA GPUs, cuFFT delivers the floating‐point performance of a GPU without having to develop your own custom GPU FFT implementation.