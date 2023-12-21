# About GPUs

## Choice of GPUs
[Which GPU(s) to Get for Deep Learning: My Experience and Advice for Using GPUs in Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) by Tim Dettmers

## Comparison of GPUs
[List of Nvidia graphics processing units](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units#Quadro_RTX_x000_series) by Wikipedia

## GPU python programming packages
### General programming
- CuPy
- Numba
- PyCUDA

### Curve fit
- [gpufit](https://github.com/gpufit/Gpufit)
- [JAXFit](https://github.com/Dipolar-Quantum-Gases/jaxfit): faster than gpufit but requires JAX, which doesn't natively support GPU on Windows. Need to install JAX in third-party ways. Instructions on JAXFit GitHub homepage. Also, mnake sure to manually install JAX, otherwise when installing JAXFit, a cpu-only version of JAX will be installed and won't take advantage of GPU acceleration.