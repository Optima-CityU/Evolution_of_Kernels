<h1 align="center">Evolution of Kernels: Automated RISC-V Kernel Optimization with Large Language Models</h1>

<div align="center">

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Optima-CityU/Evolution_of_Kernels?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/Optima-CityU/Evolution_of_Kernels)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)
![GitHub issues](https://img.shields.io/github/issues-raw/Optima-CityU/Evolution_of_Kernels)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Optima-CityU/Evolution_of_Kernels)
![GitHub](https://img.shields.io/github/license/Optima-CityU/Evolution_of_Kernels)

**IMPORTANT**: This is a very early version, and is not ready for scale use. 

We are working hard with our corporators to provide a more stable and user-friendly version. The [roadmap](#roadmap) is shown below.

</div>

## About The Project
Evolution of Kernels (EoK) is an open-source Platform leveraging Large Language Models (LLMs) for Automatic Kernel Optimization. Please refer to the [paper](https://arxiv.org/abs/2509.14265) for detailed information, including the overview, methodology, and benchmark results.


# Table of Contents

- [Table of Contents](#table-of-contents)
- [Getting Started](#getting-started)
- [Quickstart/Demo](#quickstartdemo)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contribute](#contribute)
- [Reference](#reference)
- [License](#license)

# Getting Started

# Quickstart/Demo
We are working on an easy-deployment demo as our first priority.Stay tuned!

You can check out the [roadmap](#roadmap) for more information.

## Prerequisites

You need to prepare two nodes with different IP addresses for kernel evolving and testing currently. However, we are working on a more convenient deployment method.

Additionally, we use `uv` to set up the environment. Conda or pip would be working correctly; however, we have not conducted any tests. We welcome anyone to test and provide feedback to us.

A container version is developing. However, the hardware requirements are still being tested, and existing evidences show that the containerized version for kernel evaluation can lead to large errors.

## Installation

1. Fill in the config.toml

3. Run main.py on host device ("uv run main.py")
4. After task_initializing successed, run task_hosting.py to host the task for devices parallelly evolving kernels
5. Similarly, use uv to setup the environment on devices, then run task_hosting.py on devices to evolving the kernels
6. Collect the results from devices, and run gen_report.py to generate the final report

[(Back to top)](#table-of-contents)

# Usage
[(Back to top)](#table-of-contents)

# Roadmap

- [x] Support Asynchronous operator performance evaluation
- [ ] Support Single-Device Deployment for user testing
- [ ] Provide a more stable and user-friendly deployment document
- [ ] Support More Types of Devices/Kernels 
    - [ ] Huawei Ascend / CANN
    - [ ] Nvidia GeForce / CUDA
    - [ ] AMD / ROCm
- [ ] Support Customized Methods
    - [ ] [LLM4AD](https://github.com/Optima-CityU/LLM4AD)
    - [ ] Modular Design

See the [open issues](https://github.com/Optima-CityU/Evolution_of_Kernels/issues) for a full list of proposed features (and known issues).


# Contribute
We are more than welcome to contribute including developing code and ideas to improve our platform.
+ **Issue:** If you find a bug or you have any kind of concern regarding the correctness, please report us an [issue](https://github.com/Optima-CityU/Evolution_of_Kernels/issues).
+ **Profit Purpose:** If you intend to use LLM4AD for any profit-making purposes, please contact [us](mailto:siyuan.chen@my.cityu.edu.hk).
  
[(Back to top)](#table-of-contents)

# Reference

If you find EoK helpful please cite:

```bibtex
@misc{chen2025evolutionkernelsautomatedriscv,
      title={Evolution of Kernels: Automated RISC-V Kernel Optimization with Large Language Models}, 
      author={Siyuan Chen and Zhichao Lu and Qingfu Zhang},
      year={2025},
      eprint={2509.14265},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2509.14265}, 
}
```

# License
Distributed under the MIT License. See `LICENSE` for more information.

[(Back to top)](#table-of-contents)
