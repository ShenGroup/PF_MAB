# Federated Multi-armed Bandits with Personalization

This is the supplementary code for the AISTATS 2021 proceeding "Federated Multi-armed Bandits with Personalization". The details of codes are specified as following and the results can be get by directly running the notebook "PF-UCB.ipynb". The obtained data will be stored in folders: scores, scores_improved, and socres_movielens.

## Requirement

- Python 3.7
- numpy 1.18.1
- matplotlib 3.1.3

## Code Description

The PF-UCB algorithm is implemented in files "bandits.py", "client.py" and "server.py". The enhanced version is implemented in files "bandits_improved.py", "client_improved.py" and "server_improved.py".

## Dataset Description

The synthetic dataset used in Fig. (1) is contained in the notebook. The original MovieLens dataset can be downloaded from [here(https://grouplens.org/datasets/hetrec-2011/)][here] and the preprocessing steps are specified in the paper. The processed MovieLens dataset is contained in the file "movielens_norm_10_40.npy". The synthetic dataset used in Fig. (4) is contained in file "means.npy".

