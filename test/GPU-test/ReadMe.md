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
    - only support arrays in 2D fit
- [JAXFit](https://github.com/Dipolar-Quantum-Gases/jaxfit)
    - Faster than gpufit 
    - But requires JAX, which doesn't natively support GPUs on Windows. JAXFit GitHub homepage has instructions to install JAX on Windows for GPU in a third-party way. But even with this, I can only get it work for Nvidia Quadro (workstation) and Tesla (cluster) series GPUs, but not for GeForce series. Quadro and Tesla GPUs are significantly more expensive than GeForce ones of similar specs.
    - JAXFit uses JIT from JAX to accelrate. The size of the fitted array needs to keep the same to take the full advantage of JIT, otherwise the codes will re-compile and slow down. This is mentioned in the Appendix A of their [paper](https://arxiv.org/abs/2208.12187).
- [Torchimize](https://github.com/hahnec/torchimize)
    - Not for curve fit but for general least-square minimizing. Can use for curve fit with a little extra coding.
    - Seems slower than gpufit and also in-accurate for large datasets.
- Use optimization functions in machine learning packages like PyTorch, JAX, etc.
    - They are implemented for general least-square minimizing, but not for curve fit. Need extra coding to use for curve fit purpose.
    - May not have the common Levenberg-Marquardt algorithm implemented, as it's not efficient for large datasets used in machine learning.
- Write our own curve fit module based on established packages like CuPy, JAX, etc.
    - Refer to the article below for Levenberg-Marquardt algorithm.
    - [The Levenberg-Marquardt algorithm for
nonlinear least squares curve-fitting problems](https://people.duke.edu/~hpgavin/ExperimentalSystems/lm.pdf) by Henri P. Gavin.
    - [A numpy implementation](https://github.com/abnerbog/levenberg-marquardt-method/tree/main) of the above article. Can probably be adapted in CuPy or JAX, as they are packaged as drop-in replacement for numpy on GPU.