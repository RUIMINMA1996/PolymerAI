# PolymerAI
An AI-powered package for polymer informatics

## Contents

* Polymer generator
  * generate random polymers that can be used for unsupervised learning purposes
  * as a Prior, which can be optimized by algorithms like reinforcement learning later, for polymer design purposes

## Installation

PolymerAI is written in pure Python, but it does not support installation from `pip` or `conda` at current stage. More convenient ways for installation will be built in the future.

At current stage, please download the package directly from github or use 
```bash
$ git clone https://github.com/RUIMINMA1996/PolymerAI.git
``` 
to get the package.

PolymerAI consists of several modules, and in order to run each module, please go to the module's folder and excute files following the instruction in that folder.

## Inverse polymer design

run "python agent_baseline_steptest.py" to generate polymers with high thermal conductivity.

pretrained model can be downloaded via the following link:
https://drive.google.com/file/d/1gAwUq4T5MxFXnRSqmANtvAgTv7slcoc8/view?usp=sharing

## Citing PolymerAI

To cite this repository:
```markdown
@software{PolymerAI2020github,
author = {Ruimin Ma, and Tengfei Luo},
title = {{P}olymerAI: An AI-Powered Package for Polymer Informatics},
url = {https://github.com/RUIMINMA1996/PolymerAI},
version = {1.0},
year = {2020},
}
```
