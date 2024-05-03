import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
sys.path.insert(0, "/home/gvisona/Projects/MultimodalAMR/gv_experiments")
# sys.path.insert(0, "../data_split")

import numpy as np
import os
from os.path import join, exists
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from argparse import ArgumentParser
import json
from experiments.pl_experiment import Classifier_Experiment
from data_split.data_utils import DataSplitter
from models.data_loaders import DrugResistanceDataset_Fingerprints, SampleEmbDataset
from models.classifier import Residual_AMR_Classifier
import sys
import time
from copy import deepcopy
import wandb

COMBINATIONS = [
("Staphylococcus aureus", "Ciprofloxacin"),
("Klebsiella pneumoniae", "Meropenem"),
("Klebsiella pneumoniae", "Ciprofloxacin"),
("Escherichia coli", "Tobramycin"),
("Klebsiella pneumoniae", "Tobramycin"),
("Escherichia coli", "Cefepime"),
("Escherichia coli", "Ceftriaxone"),
("Escherichia coli", "Ciprofloxacin"),
("Klebsiella pneumoniae", "Cefepime"),
("Klebsiella pneumoniae", "Ceftriaxone"),
("Staphylococcus aureus", "Fusidic acid"),
("Staphylococcus aureus", "Oxacillin")
]


def main(args):
    config = vars(args)
    seed = args.seed
    time.sleep(np.random.randint(1, 15))
    comb_idx = config["combination_idx"] 
    
    output_folder = join("outputs", args.experiment_group, args.experiment_name, str(args.seed))
    metrics_folder = join("outputs", args.experiment_group, args.experiment_name, "metrics")
    experiment_folder = join("outputs", args.experiment_group, args.experiment_name)
    root_folder = config.get("root_folder", None)
    if root_folder is not None:
        output_folder = join(root_folder, output_folder)
        metrics_folder = join(root_folder, metrics_folder)
        experiment_folder = join(root_folder, experiment_folder)
    
    if not exists(output_folder):
        os.makedirs(output_folder)

    if not exists(metrics_folder):
        os.makedirs(metrics_folder)
    print("All results will be saved in ", experiment_folder)


    driams_long_table = pd.read_csv(args.driams_long_table)

    target_species, target_drug = COMBINATIONS[comb_idx]
    print(f"Combination {comb_idx} - Target species {target_species} - Target drug {target_drug}")

        
    predictions_folder = join(experiment_folder, "predictions", f"{target_species}_{target_drug}_seed{seed}" )
    if not exists(predictions_folder):
        os.makedirs(predictions_folder)
        
    # if exists(join(predictions_folder, f"split_{split_idx}.csv")):
    #     print("\n\nExperiment already performed!\n\n")
    #     sys.exit(0)


    spectra_matrix = np.load(args.spectra_matrix)
    drugs_df = pd.read_csv(args.drugs_df, index_col=0)
    driams_long_table = driams_long_table[driams_long_table["drug"].isin(drugs_df.index)]
    dsplit = DataSplitter(driams_long_table, dataset=args.driams_dataset)

    samples_list = sorted(dsplit.long_table["sample_id"].unique())
    assert len(samples_list)==len(spectra_matrix)

    ix = (dsplit.long_table["species"]==target_species)&(dsplit.long_table["drug"]==target_drug)
    test_df = dsplit.long_table[ix]
    
    trainval_df = dsplit.long_table[~ix]
    
    data_folder = join(output_folder, "pretraining_data_splits", f"{target_species}_{target_drug}_seed{seed}")
    if not exists(data_folder):
        os.makedirs(data_folder)
    
    train_df, val_df = dsplit.baseline_train_test_split(trainval_df, test_size=0.2, random_state=args.seed)
    train_df.to_csv(join(data_folder, "train_df.csv"), index=False)
    val_df.to_csv(join(data_folder, "val_df.csv"), index=False)
    test_df.to_csv(join(data_folder, "test_df.csv"), index=False)
    
    train_dset = DrugResistanceDataset_Fingerprints(train_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
    val_dset = DrugResistanceDataset_Fingerprints(val_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
    test_dset = DrugResistanceDataset_Fingerprints(test_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])


    sorted_species = sorted(dsplit.long_table["species"].unique())
    idx2species = {i: s for i, s in enumerate(sorted_species)}
    species2idx = {s: i for i, s in idx2species.items()}

    config["n_unique_species"] = len(idx2species)


    # Save configuration
    if not exists(join(experiment_folder, "config.json")):
        del config["seed"]
        with open(join(experiment_folder, "config.json"), "w") as f:
            json.dump(config, f)



    train_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = Residual_AMR_Classifier(config)
    experiment = Classifier_Experiment(config, model)

    if not exists(join(experiment_folder, "architecture.txt")):
        with open(join(experiment_folder, "architecture.txt"), "w") as f:
            f.write(experiment.model.__repr__())



    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(output_folder, 
                "checkpoints", "{}_{}".format(target_species.replace(" ", "_"), target_drug.replace(" ", "_"))),
                                          monitor="val_loss", filename="gst-{epoch:02d}-{val_loss:.4f}", save_top_k=1)
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience
    )
    callbacks = [early_stopping_callback, checkpoint_callback]

    # tb_logger = pl_loggers.TensorBoardLogger(
    #     save_dir=join(output_folder, "logs/"))
    wandb_logger = pl_loggers.WandbLogger(project=args.experiment_name)

    print("Training..")
    trainer = pl.Trainer(devices="auto", accelerator="auto", default_root_dir=output_folder, max_epochs=args.n_epochs, callbacks=callbacks,
                         logger=wandb_logger, log_every_n_steps=3, num_sanity_val_steps=0
                        #  limit_train_batches=20, limit_val_batches=10, #limit_test_batches=5
                         )
    trainer.fit(experiment, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    print("Pretraining complete!")
    print("Best checkpoint ", checkpoint_callback.best_model_path)
    if "val_loss=0.000" in checkpoint_callback.best_model_path:
        print("Pytorch Lightning bugged out, exiting..")
        os.rmdir(os.path.join(output_folder, "checkpoints"))
        sys.exit(1)

    print("Testing..")
    test_results = trainer.test(ckpt_path="best", dataloaders=test_loader)
    with open(join(metrics_folder, "test_metrics_{}.json".format(seed)), "w") as f:
        json.dump(test_results[0], f, indent=2)

    test_df["predicted_proba"] = experiment.test_predictions
    test_df.to_csv(join(predictions_folder, f"test_pretrained_{target_species.replace(' ', '_')}_{target_drug.replace(' ', '_')}.csv"), index=False)

    print("Testing complete")



if __name__=="__main__":

    parser = ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="PretrainTestSaving")
    parser.add_argument("--experiment_group", type=str, default="TestDataSaving")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--combination_idx", type=int, default=11)
    # parser.add_argument("--split_idx", type=int, default=0)

    parser.add_argument("--driams_dataset", type=str, choices=['A', 'B', 'C', 'D'], default="A")
    parser.add_argument("--driams_long_table", type=str,
                        default="/home/gvisona/Projects/AMR_Pred/processed_data/DRIAMS_combined_long_table.csv")
    parser.add_argument("--spectra_matrix", type=str,
                        default="/home/gvisona/Projects/AMR_Pred/data/DRIAMS-A/spectra_binned_6000_all.npy")
    parser.add_argument("--drugs_df", type=str,
                        default="/home/gvisona/Projects/AMR_Pred/processed_data/drug_fingerprints.csv")
    parser.add_argument("--splits_file", type=str,
                        default="/home/gvisona/Projects/AMR_Pred/data/AMR_baseline_splits_hh.json")
    parser.add_argument("--root_folder", type=str,
                        default="/home/gvisona/Projects/AMR_Pred/saving_test")

    parser.add_argument("--drug_emb_type", type=str, default="fingerprint", choices=["fingerprint", "vae_embedding"])
    parser.add_argument("--fingerprint_class", type=str, default="morgan_1024", choices=["all", "MACCS", "morgan_512", "morgan_1024", "pubchem"])
    parser.add_argument("--fingerprint_size", type=int, default=1024)


    parser.add_argument("--n_hidden_layers", type=int, default=5)
    parser.add_argument("--conv_out_size", type=int, default=512)
    parser.add_argument("--sample_embedding_dim", type=int, default=512)
    parser.add_argument("--drug_embedding_dim", type=int, default=512)


    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    # parser.add_argument("--ft_learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    # parser.add_argument("--num_workers", type=int, default=0)
    
    args = parser.parse_args()
    args.num_workers = os.cpu_count()
    args.species_embedding_dim = 0


    main(args)
