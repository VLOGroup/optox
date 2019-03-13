# Operator to X - `optox`
## Goal
Write operators only once and use it **everywhere**. 

## Concept
Write an operator in `C++/CUDA` and generate wrappers to different languages such as `Python` and machine learning libraries such as `Tensorflow` or `Pytorch`.

`optox` provides a tensor interface to ease data transfer between host tensors `optox::HTensor` and device tensors `optox::DTensor` or any floating type and number of dimensions.
Using this interface, a operator is only written once in `C++/CUDA` and wrappers for `Python`, `Tensorflow` and `Pytorch` expose the functionality to a higher level application (e.g. iterative reconstruction, custom deep learning reconstruction, ...).

## Overview 
The source files are organized as follows:

    .
    +-- src             : `optox` library source files
    |   +-- tensor      : header only implementation of `HTensor` and `DTensor`
    |   +-- operators   : actual implementation of operator functionality
    +-- python          : python wrappers 
    +-- pytorch         : pytorch wrappers
    +-- tensorflow      : tensorflow wrappers (TODO: update)

## Install instructions

First setup the following environment variables:
- `COMPUTE_CAPABILITY` with the compute capability of your CUDA-enabled GPU [see here](https://en.wikipedia.org/wiki/CUDA)
- `CUDA_ROOT_DIR` to point to the NVidia CUDA toolkit (typically `/usr/local/cuda`)
- `CUDA_SDK_ROOT_DIR` to point to the NVidia CUDA examples (typically `/usr/local/cuda/samples`)

Note that the CUDA version used to build the `optox` library should match the version required by `Tensorflow` and/or `Pytorch`.
Thus, we recommend building both deep learning frameworks from source.

Install the `Python` dependencies using `anaconda`:
- `conda install unittest`
- `conda install pybind11`

To build the basic optox library perform the following steps:
```bash
$ mkdir build
$ cd build
$ cmake .. 
$ make install
```

### `Python` wrappers
To build the `Python` wrappers `optox` requires `pybind11` which can be installed in an anaconda environment by `conda install pybind11`.
To also build `Python` wrappers substitute the `cmake` command by:
```bash
$ cmake .. -DWITH_PYTHON=ON
```

### `Pytorch` wrappers
To build it, the `pytorch` package must be installed.
```bash
$ cmake .. -DWITH_PYTORCH=ON
```
### `Tensorflow` wrappers
To build it, the `tensorflow` package must be installed.
```bash
$ cmake .. -DWITH_TENSORFLOW=ON
```

Note to multiple combinations are supported.


## Testing

### `Python`
To perform an adjointness test of the `nabla` operator using the `Python` wrappers execute
```bash
$ python -m unittest optopy.nabla

```
If successful the output should be 
```bash
(env) ∂ python -m unittest optopy.nabla 
dtype: <class 'numpy.float64'> dim: 2 diff: 6.661338147750939e-16
.dtype: <class 'numpy.float64'> dim: 3 diff: 2.842170943040401e-14
.dtype: <class 'numpy.float32'> dim: 2 diff: 2.86102294921875e-06
.dtype: <class 'numpy.float32'> dim: 3 diff: 7.62939453125e-06
.
----------------------------------------------------------------------
Ran 4 tests in 1.099s

OK

```


### `Pytorch`
To perform a gradient test of the `activations` operators using the `Pytorch` wrappers execute
```bash
$ python -m unittest optoth.activations.act

```
If successful the output should be 
```bash
(env) ∂ python -m unittest optoth.activations.act 
grad_x: -3616.3090656 num_grad_x -3616.3090955 success: True
grad_w: 7232.6181312 num_grad_w 7232.6181312 success: True
.grad_x: 535.2185935 num_grad_x 535.2185935 success: True
grad_w: 2236.8791233 num_grad_w 2236.8791233 success: True
.grad_x: -215.0009414 num_grad_x -215.0009432 success: True
grad_w: 430.0018828 num_grad_w 430.0018828 success: True
.
----------------------------------------------------------------------
Ran 3 tests in 2.263s

OK

```