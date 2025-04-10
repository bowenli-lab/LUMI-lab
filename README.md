# LUMI-lab

[![bioRxiv](https://img.shields.io/badge/Preprint-bioRxiv-green)](https://www.biorxiv.org/content/10.1101/2025.02.14.638383)
[![YouTube](https://img.shields.io/badge/Demo-YouTube-red)](https://youtu.be/POOgIiKRSiE)

Foundation model-driven lab enabling discovery of ionizable lipid design

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

LUMI-lab is a foundation model-driven laboratory designed to enable the discovery and design of ionizable lipids. This project integrates various components, including 3D printing models, control panels, and automated systems, to facilitate advanced lipid research.

## Usage
Check the following directory for details!


| Directory | Description |
| ------- | ----------- |
| [3D Printing](/3D_printing_models/README.md) | The directory includes 3D printing models for various components. |
| [Control Panel](/control_panel/README.md) | The control panel directory contains source code the for orchestration system and GUI for controlling and monitoring the lab. |
| [Opentrons](/opentron/README.md) | The opentron directory contains the protocols for the Opentrons robot, and config files for customized labwares, |
| [Model](/model/README.md) | The model directory contains the foundation model and the training code. |



## Core Contributors

[Haotian Cui](https://github.com/subercui) ([subercui@gmail.com](mailto:subercui@gmail.com)), [Kuan Pang](https://github.com/Kuan-Pang), [Gen Li](https://github.com/ReaganGen), [Yue Xu](https://github.com/cpuxuyue)

## License

This project is licensed under the terms of the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

We would like to express special thanks to the following open-source projects, which have been instrumental in the development of LUMI-lab:

- [Uni-Mol](https://github.com/deepmodeling/Uni-Mol/tree/main)
- [Uni-Core](https://github.com/dptech-corp/Uni-Core)
- [transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://github.com/pytorch/pytorch)
- [RDKiT](https://github.com/rdkit/rdkit)
- [flash-attention](https://github.com/Dao-AILab/flash-attention)
- [Alab Management](https://github.com/CederGroupHub/alabos)


## Citing LUMI-lab
```
@article{cui2025lumi,
  title={LUMI-lab: a Foundation Model-Driven Autonomous Platform Enabling Discovery of New Ionizable Lipid Designs for mRNA Delivery},
  author={Cui, Haotian and Xu, Yue and Pang, Kuan and Li, Gen and Gong, Fanglin and Wang, Bo and Li, Bowen},
  journal={bioRxiv},
  pages={2025--02},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
