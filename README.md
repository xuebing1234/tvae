# tVAE
This is the implementation of the tVAE model presented for KDD submission: https://openreview.net/pdf?id=tfSUbSijOqZ

The model is tailored for treatment effect prediction for ECMO predictions, and is validated by the public IHDP dataset. Here we attached the notebook with outputs to demonstrate the results reported in paper. 

We also notice the inconsistencies of existing papers when using the IHDP dataset, by manipulating the sampling strategy and/or performance metrics. Here we also share the results using the proposed model in these papers, standardized by the same train/test splits for 1000 replications and evaluated by the same rPEHE metric. 
