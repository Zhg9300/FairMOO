# FairMOO: Achieving Fairness in Distributed Learning via Constrained Multi-Objective Optimization
This hub stores the code for paper FairMOO: Achieving Fairness in Distributed Learning via Constrained Multi-Objective Optimization

## Reference & Acknowledgement

This project incorporates partial implementations from the following work:

> Zibin Pan, et al., "FedLF: Layer-Wise Fair Federated Learning", *AAAI*, 2024.

Code: [https://github.com/zibinpan/FedLF](https://github.com/zibinpan/FedLF)

We thank the authors for making their code publicly available, which inspired parts of our implementation.

## Requirements

The code can be run under **Python 3.9** with dependencies below:

```
numpy==1.25.1
pytorch==2.0.1
torchvision==0.15.2
cvxopt==1.3.2
```

## Quick Start

Set the current folder as `./FairMethods`, and then run `run.py`. Here we provide FairMOO as an example.
```
python run.py --device 0 --module CNN --algorithm FairMOO --prefer 0.003 --dataloader DataLoader_cifar10_pat --N 10 --NC 1 --B 256 --R 3000 --lr 0.2 --decay 0.999 --test_interval 50
```
To reproduce results for other fair algorithms, simply replace `FairMOO` with other algorithm names (e.g., FedAvg, AdaFed, FedMGDA_plus, qFedAvg) in the `--algorithm` argument. All parameters can be seen in `./FairMethods/main.py`.

## Detailed Numerical Results in the Paper

Here we provide the detailed numerical results in our paper.
### MLP on FashionMNIST

| Dir($\alpha = 0$) | | | | Dir($\alpha = 0.1$) | | | |
|---------|-|-|-|---------|-|-|-|
| **Algorithm** | acc(%)↑ | var(%)↓ | worst(%)↑ | **Algorithm** | acc(%)↑ | var(%)↓ | worst(%)↑ |
| FedAvg | 88.20 | 87.26 | 66.6 | FedAvg | 88.76 | 51.54 | 71.48 |
| qFFL | 89.02 | 64.02 | 75.5 | qFFL | 89.83 | 47.95 | 74.74 |
| DRFL | 88.04 | 67.22 | 71.0 | DRFL | 88.46 | 45.90 | 74.31 |
| AFL | 84.93 | 15.43 | 79.0 | AFL | 86.38 | 23.13 | 80.07 |
| FedMGDA+ | 89.90 | 91.16 | 66.8 | FedMGDA+ | 87.56 | 82.26 | 67.54 |
| AdaFed | 89.32 | 60.05 | 72.5 | AdaFed | 90.19 | 29.76 | 80.93 |
| FedMDFG | 86.59 | 39.54 | 70.8 | FedMDFG | 88.56 | 33.28 | 74.40 |
| FedLF | 88.56 | 50.44 | 74.8 | FedLF | 89.65 | 27.65 | 80.07 |
| **FairMOO** | 86.07 | 32.08 | 78.1 | **FairMOO** | 87.65 | 25.33 | 80.16 |

### CNN on CIFAR10

| Dir($\alpha = 0$) | | | | Dir($\alpha = 0.1$) | | | |
|---------|-|-|-|---------|-|-|-|
| **Algorithm** | acc(%)↑ | var(%)↓ | worst(%)↑ | **Algorithm** | acc(%)↑ | var(%)↓ | worst(%)↑ |
| FedAvg   | 74.41 | 110.75 | 55.8  | FedAvg   | 68.3  | 37.70  | 60.30  |
| qFFL     | 75.05 | 99.50   | 56.5  | qFFL     | 67.22 | 39.72 | 58.54 |
| DRFL     | 74.24 | 79.54  | 58.5  | DRFL     | 66.99 | 31.29 | 59.33 |
| AFL      | 71.39 | 33.06  | 62.7  | AFL      | 67.61 | 31.11 | 58.40  |
| FedMGDA+ | 74.47 | 94.29  | 58.3  | FedMGDA+ | 67.66 | 45.93 | 55.80  |
| AdaFed   | 70.47 | 54.70   | 59.3  | AdaFed   | 66.9  | 55.90  | 55.80  |
| FedMDFG  | 74.43 | 77.18  | 58.5  | FedMDFG  | 64.88 | 36.79 | 57.29 |
| FedLF | 74.78 | 55.80 | 63.0 | FedLF | 64.78 | 31.52 | 53.78 |
| **FairMOO**   | 71.23 | 44.89  | 60.4  | **FairMOO**  | 66.70 | 33.24 | 57.98 |

