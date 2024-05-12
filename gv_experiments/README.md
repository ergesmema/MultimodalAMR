# ResMLP training

This folder contains the experiments performed by G.V.
The scripts were run on Python 3.9.16, with the packages contained in the `requirements.txt` file.

The `training_scripts` folder contains the code used to produce the results presented in the paper.
Specifically:

- `train_ResAMR_classifier.py` contains the code to train a ResMLP model on the DRIAMS datasets for the predictions tasks.
- `train_recommender_ResAMR.py` is the script used to train the model for recommendation purposes.
- `train_SplitPCA_LR_classifier.py` and `train_JointPCA_LR_classifier.py` contain the training of the PCA+LR baseline. They respectively apply the PCA step before and after concatenation.
- `train_1hotSpecies_ResAMR_classifier.py` is the ablation experiment where the MALDI-TOF spectrum is substituted by the 1-hot encoding of the pathogen species.
- `pretrain_ResAMR_baseline_comparison.py` and `finetune_ResAMR_baseline_comparison.py` are used for the comparison with the single drug-single species from the Weis et al. paper.

### May 2024 - XAIML
To run the training script:
- First change the DATA_PATH with the absolute value of the directory where you have the driam datasets.
- For a clean run, make sure you have deleted anything under `/output` and `/binned_data`.
- Command to run the training (once you are in gv_experiments):

```bash
python3 training_scripts_may_24/train_ResAMR_classifier_may_24.py \
--experiment_name "myExperiment1" \
--experiment_group "ResMLP" \
--seed 0 \
--split_type "random" \
--driams_long_table "../processed_data/DRIAMS_combined_long_table.csv" \
--drugs_df "../processed_data/drug_fingerprints.csv" \
--spectra_matrix "../data/spectra_binned_6000_all.npy" \
--n_epochs 2 \
--learning_rate 0.0003 \
--drug_emb_type "fingerprint" \
--fingerprint_class "morgan_1024" \
--fingerprint_size 1024 \
--patience 50 \
--batch_size 128
```


To run inference:
- Define the paths at the top of `inference.py`. For now, paths are static for testing purposes.
- Command to run inference (once you are in gv_experiments):
`python3 inference_may_24/inference.py`



