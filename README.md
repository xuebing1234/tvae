# tVAE
This is the implementation of the tVAE model submitted to KDD ADS 2023: https://openreview.net/pdf?id=tfSUbSijOqZ

The model is tailored for treatment effect prediction for COVID-19 ECMO patients, and is also validated by the public IHDP dataset. Here we attach the notebook with outputs to demonstrate the results reported in paper. 

We also notice the inconsistencies of existing papers when using the IHDP dataset, by manipulating the sampling strategy and/or performance metrics. Here we also replicate the proposed model in these papers, standardized by the same train/test splits for 1000 replications and evaluated by the same rPEHE metric. 

Besides the IHDP results, we also upload a demo of the distribution balancing effect in real and public ISARIC dataset. Let me know if you have any questions or needs. 
