---
title: The R Parallel Programming Blog
tags:
  - machine learing
  - parallel computing
  - performance optimization
  - R
  - rstats
url: 140.html
id: 140
categories:
  - Accelerators
  - General
  - MPI
  - MultiCores
  - Performance Optimizaiton
  - Vectorization
date: 2016-02-02 22:53:13
---

* * *

DISCLAIMER
----------

_This is a personal weblog. The opinions expressed here represent my own and not those of my employer. **Further,** the opinions expressed by the ParallelR Bloggers and those providing comments are theirs alone and do not reflect the opinions of  [ParallelR](http://parallelr.com)._

* * *

Today, [parallel computing](https://en.wikipedia.org/wiki/Parallel_computing) truly is a mainstream technology. But, [stock R](http://www.r-project.org/) is still a single-thread and main memory (RAM) limited software, which really restricts its usage and efficiency against the challenges from very complex model architectures, dynamically configurable analytics models and big data input with billions of parameters and samples.

Therefore, [ParallelR](http://www.parallelr.com) dedicated on accelerate R by parallel technologies, and our blog will deliver massive parallel technologies and programming tips with real cases in Machine Learning, Data Analysis, Finance fields. And we will cover rich of topics from data vectorization, usages of parallel packages, ([snow](https://cran.r-project.org/web/packages/snow/index.html), [doparallel](https://cran.r-project.org/web/packages/doParallel/index.html), [Rmpi](https://cran.r-project.org/web/packages/Rmpi/index.html), [SparkR](https://spark.apache.org/docs/1.5.2/sparkr.html)) , to parallel algorithm design and implementation by [OpenMP](http://www.openmp.org/), [OpenACC](http://www.openacc.org/), [CPU/GPU accelerated libraries](http://developer.nvidia.com/gpu-accelerated-libraries), [CUDA C/C++](http://www.nvidia.com/object/cuda_home_new.html) and [Pthread](https://en.wikipedia.org/wiki/POSIX_Threads) in R.

At [ParallelR Blog](http://www.parallelr.com/blog/) you will find useful information about productive, high-performance programming techniques based on commercialized computer architecture, ranging from multicores CPU, GPU, [Intel Xeon Phi](http://www.intel.com/content/www/us/en/processors/xeon/xeon-phi-detail.html), [FPGA](https://en.wikipedia.org/wiki/Field-programmable_gate_array) to HPC Cluster. As well, you will learn how you can use your existing skills in R in new ways, represented your R codes with structured computational models.

ParallelR Blog is created by Peng Zhao. Peng have rich experience in heterogeneous and parallel computing areas including multi-cores, multi-nodes and accelerators (GPGPU, Intel Xeon Phi) for parallel algorithm design, implementation, debugging and optimizations.

 
This is Peng. Handsome, Right?
[![](http://www.parallelr.com/wp-content/uploads/2016/02/PengZhao@ParallelR.png)](http://www.parallelr.com/wp-content/uploads/2016/02/PengZhao@ParallelR.png)