# About curve fit on GPUs

## Choice of GPUs
[Which GPU(s) to Get for Deep Learning: My Experience and Advice for Using GPUs in Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) by Tim Dettmers

## Comparison of GPUs
[List of Nvidia graphics processing units](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units#Quadro_RTX_x000_series) by Wikipedia

## GPU python programming packages
### General programming
- CuPy
- Numba
- PyCUDA
- PyTorch
- Tensorflow
- JAX

### Curve fit
- [gpufit](https://github.com/gpufit/Gpufit)
    - only support square arrays in 2D fit
- [JAXFit](https://github.com/Dipolar-Quantum-Gases/jaxfit)
    - Faster than gpufit 
    - But requires JAX, which doesn't natively support GPUs on Windows. JAXFit GitHub homepage has instructions to install JAX on Windows for GPU in a third-party way. But even with this, I can only get it work for Nvidia Quadro (workstation) and Tesla (cluster) series GPUs, but not for GeForce series. Quadro and Tesla GPUs are significantly more expensive than GeForce ones of similar specs.
    - JAXFit uses JIT from JAX to accelrate. The size of the fitted array needs to keep the same to take the full advantage of JIT, otherwise the codes will re-compile and slow down. This is mentioned in the Appendix A of their [paper](https://arxiv.org/abs/2208.12187).
- [Torchimize](https://github.com/hahnec/torchimize)
    - Not for curve fit but for general least-square minimizing. Can use for curve fit with a little extra coding.
- Use optimization functions in machine learning packages like PyTorch, JAX, etc.
    - They are implemented for general least-square minimizing, but not for curve fit. Need extra coding to use for curve fit purpose.
    - May not have the common Levenberg-Marquardt algorithm implemented, as it's not efficient for large datasets used in machine learning.
- Write our own curve fit module based on established packages like CuPy, JAX, etc.
    - Refer to the article below for Levenberg-Marquardt algorithm.
    - [The Levenberg-Marquardt algorithm for
nonlinear least squares curve-fitting problems](https://people.duke.edu/~hpgavin/ExperimentalSystems/lm.pdf) by Henri P. Gavin.
    - [A numpy implementation](https://github.com/abnerbog/levenberg-marquardt-method/tree/main) of the above article. Can probably be adapted in CuPy or JAX, as they are packaged as drop-in replacement for numpy on GPU.
- **My final choice**
    - Modify Torchimize package by explicitly implementling the Jacobian matrix in code.

## Package installation (for PyTorch and CuPy)
1. Microsoft Visual Studio with C++ build tools.
    - https://visualstudio.microsoft.com/downloads/
    - By the time of writing, version 2022 community is used, and option 'Desktop development with C++' is selected during installation.

2. Nvidia CUDA toolkit.
    - https://developer.nvidia.com/cuda-toolkit-archive
    - By the time of writing, version 12.1 is used.

3. Nvidia cuDNN library.
    - https://developer.nvidia.com/rdp/cudnn-archive
    - By the time of writing, version 8.9.6 is used. 
    - Follow the installation guide below to move the files to CUDA toolkit folder *C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDNN\\v8.9\\*,
    and add the path to *bin\\* to the system environment variable PATH.
    - https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows

4. PyTorch
    - https://pytorch.org/get-started/locally/
    - Make sure to select a CUDA version, instead of CPU-only version.
    - By the time of writing, version 2.1.2 is used.

5. CuPy
    - https://docs.cupy.dev/en/stable/install.html#installing-cupy
    - Choose the version based on installed CUDA vesion.