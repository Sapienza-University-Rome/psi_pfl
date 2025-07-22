# PSI-PFL: Population Stability Index for Client Selection in non-IID Personalized Federated Learning

[![paper](https://img.shields.io/badge/PAPER-arXiv-yellowgreen?style=for-the-badge)]()
&nbsp;&nbsp;&nbsp;

This is the code of the paper [PSI-PFL: Population Stability Index for Client Selection in non-IID Personalized Federated Learning]().


The code trains the PSI-PFL method and all the other baselines provided to tackle non-IID data in Federated Learning (FL). Specifically, we implement 12 FL algorithms (PSI_PFL, FedAvg, FedProx, FedAvgM, FedAdagrad, FedYogi, FedAdam, Power-Of-Choice (PoC), HACCS, FedCLS, Clustered-FL (CFL), and FedSoft) and 4 datasets (acs_income, dutch, celeba, and sent140).

## Environment setup

The following instructions aim to help with the environment setup to execute the training. It is done using Anaconda in Linux, but it can be extended to any other Python distributor and operating system.

1. **Clone the repository (or download the .zip folder)**:
   ```bash
   git clone https://github.com/Sapienza-University-Rome/psi_pfl.git
   cd psi_pfl

2. **Create and activate the conda environment (this may take several minutes)**:
   ```bash
    conda env create -f environment.yml --verbose
    conda activate psi_pfl_py310

## Usage
The following code executes the training using the default parameters:
```
python main.py
```

# Configuration Parameters

| Parameter | Description | Possible Values | Default Value |
|-----------|-------------|-----------------|---------------|
| `--root-path` | The root path for the application | Any valid path string | `get_default_path()` |
| `--dataset` | Dataset to use for training | `'acs_income', 'dutch', 'sent140', 'celeba'` | `'acs_income'` |
| `--partitioner` | Partition protocol to split (federate) data | `'dirichlet'` | `'dirichlet'` |
| `--non-iid-param` | Non-IID parameter (alpha for Dirichlet) | Float values from 0 to inf | `0.7` |
| `--num-clients` | Number of clients | Positive integers | `10` |
| `--rand-state` | Random state for reproducibility | Any integer | `42` |
| `--agg-method` | Aggregation algorithm to train | `'psi_pfl', 'fedavg', 'fedprox', 'fedavgm', 'fedadagrad', 'fedyogi', 'fedadam', 'poc', 'haccs' , 'fedcls', 'cfl', 'fedsoft'` | `'psi_pfl'` |
| `--seeds-list` | List of random seeds to use for training | Comma-separated integers | `'0,1,2,3,42'` |
| `--local-epochs` | Number of local epochs (per client) | Positive integers | `2` |
| `--comm-rounds` | Number of communication rounds in FL training | Positive integers | `10` |
| `--psi-ths-list` | List of percentiles for psi thresholds (tau) | Comma-separated numbers | `'10,20'` |
| `--lr` | Learning rate | Positive floats | `0.001` |
| `--mu-ths-list` | List of mu thresholds for fedprox | Comma-separated numbers | `'0.01'` |
| `--momentum-ths-list` | List of momentum thresholds for fedavgm | Comma-separated numbers | `'0.7'` |
| `--tau-ths-list` | List of tau thresholds for fedadagrad, fedyogi and fedadam | Comma-separated numbers | `'0.1'` |
| `--eta-ths-list` | List of eta thresholds for fedadagrad, fedyogi and fedadam | Comma-separated numbers | `'0.3162'` |
| `--eta-l-ths-list` | List of eta_l thresholds for fedadagrad, fedyogi and fedadam | Comma-separated numbers | `'1'` |
| `--beta-1-ths-list` | List of beta1 thresholds for fedyogi and fedadam | Comma-separated numbers | `'0.9'` |
| `--beta-2-ths-list` | List of beta2 thresholds for fedyogi and fedadam | Comma-separated numbers | `'0.99'` |
| `--d-poc` | Number of clients to select for power-of-choice (poc) | Positive integers | `10` |
| `--rho-haccs` | Rho threshold (for haccs) | Float between 0 and 1 | `0.95` |
| `--thl-fedcls` | Similarity threshold (for fedcls) | Float values | `0.1` |


**Download missing datasets (only if you are training with CelebA or Sent140):**

Due to GitHub's file size limitations, the CelebA and Sent140 datasets are not included in the repository. To use these datasets, you must manually download them and place the files into the appropriate folder created during the Environment Setup step.
Click on [this link](https://drive.google.com/drive/folders/14_5T_fiQgVDoFt8vXoombLjkF34o5stC?usp=sharing) to download the data.

## Citation
If you find this repository useful, please cite our paper:

```
@misc{yourarxivid,
  author  = {Daniel M. Jimenez-Gutierrez and David Solans and Mohammed Elbamby and Nicolas Kourtellis}
},
  title   = {PSI-PFL: Population Stability Index for Client Selection in non-IID Personalized Federated Learning},
  journal = {arXiv preprint},
  volume  = {arXiv:XXXX.XXXXX},
  year    = {2023},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```
