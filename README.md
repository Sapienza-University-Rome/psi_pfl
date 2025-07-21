# PSI-PFL: Population Stability Index for Client Selection in non-IID Personalized Federated Learning

[![paper](https://img.shields.io/badge/PAPER-arXiv-yellowgreen?style=for-the-badge)](https://arxiv.org/pdf/2102.02079.pdf)
&nbsp;&nbsp;&nbsp;

This is the code of the paper [PSI-PFL: Population Stability Index for Client Selection in non-IID Personalized Federated Learning]().


The code trains the PSI-PFL method and all the other baselines provided to tackle non-IID data in Federated Learning (FL). Specifically, we implement 12 FL algorithms (PSI_PFL, FedAvg, FedProx, FedAvgM, FedAdagrad, FedYogi, FedAdam, Power-Of-Choice (PoC), HACCS, FedCLS, Clustered-FL (CFL), and FedSoft) and 4 datasets (acs_income, dutch, celeba, and sent140).

## Environment setup

The following instructions aim to help with the environment setup to execute the training. It is done using Anaconda, but it can be extended to any other Python distributor.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sapienza-University-Rome/psi_pfl.git
   cd psi_pfl

2. **Create and activate the conda environment**:
   ```bash
    conda name_for_env create -f environment.yml
    conda activate name_for_env

2. **Install required libraries**:
   ```bash
   conda

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
| `--non-iid-param` | Non-IID parameter (alpha for Dirichlet) | Float values from 0 to inf | `1` |
| `--num-clients` | Number of clients | Positive integers | `10` |
| `--rand-state` | Random state for reproducibility | Any integer | `42` |
| `--agg-method` | Aggregation algorithm to train | `'psi_pfl', 'fedavg', 'fedprox', 'fedavgm', 'fedadagrad', 'fedyogi', 'fedadam', 'poc', 'haccs' , 'fedcls', 'cfl', 'fedsoft'` | `'psi_pfl'` |
| `--seeds-list` | List of random seeds to use for training | Comma-separated integers | `'42,0'` |
| `--local-epochs` | Number of local epochs (per client) | Positive integers | `2` |
| `--comm-rounds` | Number of communication rounds in FL training | Positive integers | `2` |
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

3. **Download missing datasets (only if you are training with CelebA or Sent140):**

   * For sent140:
   ```bash
    cd data/sent140
    wget https://drive.google.com/file/d/1VkpPeWV1sAv-JxWWJn0D7KLMWbS8yhDj/view?usp=drive_link -O training.1600000.processed.noemoticon.zip
    unzip training.1600000.processed.noemoticon.zip
    rm training.1600000.processed.noemoticon.zip
    cd glove_6B
    wget https://drive.google.com/file/d/1BWJMhQnJLg2bGlG5ONJReYmrm0kgjgrj/view?usp=drive_link -O glove.6B.300d.zip
    unzip glove.6B.300d.zip
    rm glove.6B.300d.zip
    cd ../../..
    ```
   * For celeba:
   ```bash
    cd data/celeba
    wget https://drive.google.com/file/d/1xVLmWtukUkb9L2ry7hzQnPGTePxHwH-x/view?usp=drive_link -O img_align_celeba.zip
    unzip img_align_celeba.zip
    rm img_align_celeba.zip
    cd ../..
    ```

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
