# Parallel aBUS

[![PyPI - Version](https://img.shields.io/pypi/v/parallel-abus.svg)](https://pypi.org/project/parallel-abus)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/parallel-abus.svg)](https://pypi.org/project/parallel-abus)

<!-- [![DOI](https://zenodo.org/badge/357505574.svg)](https://zenodo.org/badge/latestdoi/357505574) -->

-----

**Table of Contents**

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

This code is a Python implementation of the parallelized adaptive Bayesian Updating with Structural reliabilty methods. The methods were described in:

> Simon, P., Schneider, R., Baeßler, M., Morgenthal, G., 2024. Parallelized adaptive Bayesian Updating with Structural reliability methods for inference of large engineering models (submitted for publication).

It is based upon the [original Python implementation](https://www.cee.ed.tum.de/era/software/bayesian/abus/) of the original adaptive Bayesian Updating with Structural reliability methods with Subset Simulation (aBUS-SuS) by the Engineering Risk Analysis (ERA) Group of Technische Universität München.


## Installation

The package is installable from the Python Package Index (PyPI) using `pip`:

```console
pip install parallel-abus
```

## Usage

Usage is exemplified in the corresponding GitHub project of this package.

Examples using this package are documented in the `./tests/` folder. The number of processes can be specified as a command line parameter, for example:

```console
python ./tests/test_main_example_3_2DOF.py 5
```

Runs inference with parallel aBUS on 5 processes.


A more comprehensive example is presented in `./example/bayesian_inference.py`. Here, an engineering model of a reinforced concrete beam including an OpenSees finite element model is updated. Details on this example are found in this contribution:

> Simon, P., Schneider, R., Baeßler, M., Morgenthal, G., 2024. A Bayesian probabilistic framework for building models for structural health monitoring of structures subject to environmental variability. (submitted for publication).

This example requires amongst others the [python package for OpenSees](https://openseespydoc.readthedocs.io/en/latest/index.html).

An easy way to get this example running is to install its dependencies via [Poetry](https://python-poetry.org/):

```console
poetry install
```

## License

`parallel-abus` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
