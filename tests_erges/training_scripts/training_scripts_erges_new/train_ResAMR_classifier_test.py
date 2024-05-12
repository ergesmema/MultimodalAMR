import sys
sys.path.insert(0, "..")
sys.path.insert(0, "../../data_split")
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
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
from multimodal_amr_tests_erges.experiments.pl_experiment import Classifier_Experiment
import itertools
from multimodal_amr_tests_erges.data_split.data_utils import DataSplitter
from multimodal_amr_tests_erges.models.data_loaders import DrugResistanceDataset_Fingerprints, SampleEmbDataset, DrugResistanceDataset_Embeddings
from multimodal_amr_tests_erges.models.classifier import Residual_AMR_Classifier

import shap


import shutil  # Import shutil for copying files


TRAINING_SETUPS = list(itertools.product(['A', 'B', 'C', 'D'], ["random", "drug_species_zero_shot", "drugs_zero_shot"], np.arange(5)))

SPECIES_ANTIBIOTIKA = [
                # Bacteria species
                'Escherichia coli', 
                'Klebsiella pneumoniae', 
                'Klebsiella oxytoca', 

                'Enterobacter cloacae',
                'MIX!Enterobacter cloacae',

                'Enterobacter asburiae',
                "Enterobacter ludwigii",
                "Enterobacter aerogenes",
                "Enterobacter kobei",
                "MIX!Enterobacter asburiae",
                "MIX!Enterobacter kobei",
                "Enterobacter hormaechei",
                "Enterobacter cancerogenus",
                "MIX!Enterobacter ludwigii",
                "MIX!Enterobacter aerogenes",

                "Serratia marcescens",
                "MIX!Serratia marcescens",
                "Serratia liquefaciens",
                "Serratia ureilytica",
                "Serratia grimesii",
                "Serratia ficaria",
                "Serratia proteamaculans",
                "Serratia rubidaea",
                "Serratia fonticola",
                "MIX!Serratia liquefaciens",
                "Serratia odorifera"  

                'Proteus mirabilis', 
                'Proteus vulgaris', 
                'Morganella morganii',
                'Citrobacter freundii', 
                'Citrobacter koseri',
                'Pseudomonas aeruginosa', 
                'Stenotrophomonas maltophilia',
                'Acinetobacter baumannii',
                
                #Fungi Species
                'Staphylococcus aureus',
                'Methicillin-resistenter S. aureus',
                'Koagulase-negative Staphylokokken',
                "Enterococcus dispar",
                "Enterococcus avium",
                "MIX!Enterococcus avium",
                "Enterococcus raffinosus",
                "Enterococcus hirae",
                "Enterococcus casseliflavus",
                "Enterococcus durans",
                "MIX!Enterococcus gallinarum",
                "Enterococcus gallinarum",
                "MIX!Enterococcus casseliflavus",
                "MIX!Enterococcus mundtii",
                "Enterococcus pseudoavium",
                "MIX!Enterococcus hirae",
                "Enterococcus gilvus",
                "Enterococcus mundtii",
                "Enterococcus malodoratus",
                "Enterococcus faecalis",
                "MIX!Enterococcus faecalis",
                "Enterococcus faecium",
                "MIX!Enterococcus faecium",
                "Streptococcus pneumoniae"

]

DRUGS_ANTIBIOTIKA = [
    "Ampicillin", "Amoxicillin", "Piperacillin", "Cefuroxim", "Cefepim", "Ceftriaxon", "Meropenem", "Gentamicin", "Cotrimoxazol", "Ciprofloxacin", "Cefoxitim", "Penicillin", "Gentamicin", "Cotrimoxazol", "Makrolide", "Clindamycin",
    "Rifampicin", "Teicoplanin", "Vancomycin", "Oxacillin",
]


def filter_from_antibiotika(driams_long_table):
    # Convert 'specia' to lowercase for uniform comparison
    species = [s.lower() for s in SPECIES_ANTIBIOTIKA]
    drugs = [s.lower() for s in DRUGS_ANTIBIOTIKA]


    # Filter using a lambda function to check if any substring from 'specia' is in the 'species' column values
    species_filtered_df = driams_long_table[driams_long_table['species'].str.lower().apply(lambda x: any(s in x for s in species))]
    drugs_species_filtered_df = species_filtered_df[species_filtered_df['drug'].str.lower().apply(lambda x: any(s in x for s in drugs))]
    unique_df = drugs_species_filtered_df.drop_duplicates()
    return unique_df

def extract_binned_data(filtered_df):
    # Base directory for the binned data
    base_path = "/Users/em/Desktop/Uni-Spring24/XAIML/MultimodalAMR/"
    target_dir = 'binned_data'
    datasets = ['A', 'B', 'C', 'D']

    sample_ids = []
    spectra_matrix = []

    # Create the base directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
        # Create subdirectories A, B, C, D if they don't exist
        for dataset in datasets:
            dir_path = os.path.join(target_dir, dataset)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        for index, row in filtered_df.iterrows():
            dataset = row['dataset']
            sample_id = row['sample_id']
            file_name = f"{sample_id}.txt"
            years = ['2018'] if dataset != 'A' else ['2015', '2016', '2017', '2018']

            for year in years:
                source_dir = f"DRIAMS-{dataset}/binned_6000/{year}"
                # Path where the file is supposed to be (example given as below the dataset directory)
                source_path = os.path.join(base_path, source_dir, file_name)
                
                # Destination path (assumed requirement: move to directory above under the dataset name)
                destination_path = os.path.join(target_dir, dataset, f"{file_name}")
                
                # Move the file to the desired location
                if os.path.exists(source_path):
                    shutil.copy(source_path, destination_path)
                    sample_data = pd.read_csv(source_path, sep=" ", index_col=0)
                    spectra_matrix.append(sample_data["binned_intensity"].values)
                    
                    sample_ids.append((sample_id, year))
                else:
                    pass
                    # print(f"File {source_path} does not exist.")
        spectra_matrix = np.vstack(spectra_matrix)
        np.save("spectra_binned_6000_all.npy", spectra_matrix)
        # Convert the list to a DataFrame
        df = pd.DataFrame(sample_ids, columns=['sample_id', 'year'])
        # Save the DataFrame to a CSV file
        df.to_csv('existing_sample_ids.csv', index=False)


def main(args): #here
    config = vars(args)
    seed = args.seed
    # Setup output folders to save results
    output_folder = join(args.output_folder, args.experiment_group, args.experiment_name, str(args.seed))
    if not exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    results_folder = join(args.output_folder, args.experiment_group, args.experiment_name + "_results")
    if not exists(results_folder):
        os.makedirs(results_folder, exist_ok=True)

    experiment_folder = join(args.output_folder, args.experiment_group, args.experiment_name)
    if exists(join(results_folder, f"test_metrics_{args.seed}.json")):
        sys.exit(0)
    if not exists(experiment_folder):
        os.makedirs(experiment_folder, exist_ok=True)

    # Read data
    driams_long_table = pd.read_csv(args.driams_long_table)
    antibiotika_filtered_long_table = filter_from_antibiotika(driams_long_table)
    extract_binned_data(antibiotika_filtered_long_table)
    existing_sample_ids = pd.read_csv("existing_sample_ids.csv")
    antibiotika_existing_filtered_long_table = pd.merge(antibiotika_filtered_long_table, existing_sample_ids, on="sample_id", how="inner")

    spectra_matrix = np.load(args.spectra_matrix).astype(float)
    drugs_df = pd.read_csv(args.drugs_df, index_col=0)
    antibiotika_existing_filtered_long_table = antibiotika_existing_filtered_long_table[antibiotika_existing_filtered_long_table["drug"].isin(drugs_df.index)]
    
    # Instantate data split
    dsplit = DataSplitter(antibiotika_existing_filtered_long_table, dataset=args.driams_dataset)
    samples_list = sorted(dsplit.long_table["sample_id"].unique())

    other_metadata = ...

    # Split selection for the different experiments.
    if args.split_type == "random":
        train_df, val_df, test_df = dsplit.random_train_val_test_split(val_size=0.1, test_size=0.2, random_state=args.seed)
    elif args.split_type =="drug_species_zero_shot":
        trainval_df, test_df = dsplit.combination_train_test_split(dsplit.long_table, test_size=0.2, random_state=args.seed)
        train_df, val_df = dsplit.baseline_train_test_split(trainval_df, test_size=0.2, random_state=args.seed)
    elif args.split_type =="stratify_species": 
        train_df, val_df, test_df = dsplit.stratified_train_val_test_split(val_size=0.1, test_size=0.2, random_state=args.seed)
    elif args.split_type =="drugs_zero_shot":
        drugs_list = sorted(dsplit.long_table["drug"].unique())
        if args.seed>=len(drugs_list):
            print("Drug index out of bound, exiting..\n\n")
            sys.exit(0)
        target_drug = drugs_list[args.seed]
        # target_drug = args.drug_name
        test_df, trainval_df = dsplit.drug_zero_shot_split(drug=target_drug)
        train_df, val_df = dsplit.baseline_train_test_split(trainval_df, test_size=0.2, random_state=args.seed)

    test_df.to_csv(join(output_folder, "test_set_stratified_species.csv"), index=False)

    if args.drug_emb_type=="fingerprint":
        train_dset = DrugResistanceDataset_Fingerprints(train_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
        val_dset = DrugResistanceDataset_Fingerprints(val_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
        test_dset = DrugResistanceDataset_Fingerprints(test_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
    elif args.drug_emb_type=="vae_embedding" or args.drug_emb_type=="gnn_embedding":
        train_dset = DrugResistanceDataset_Embeddings(train_df, spectra_matrix, drugs_df, samples_list)
        val_dset = DrugResistanceDataset_Embeddings(val_df, spectra_matrix, drugs_df, samples_list)
        test_dset = DrugResistanceDataset_Embeddings(test_df, spectra_matrix, drugs_df, samples_list)


    sorted_species = sorted(dsplit.long_table["species"].unique())
    idx2species = {i: s for i, s in enumerate(sorted_species)}
    species2idx = {s: i for i, s in idx2species.items()}

    config["n_unique_species"] = len(idx2species)
    del config["seed"]
    # Save configuration
    if not exists(join(experiment_folder, "config.json")):
        with open(join(experiment_folder, "config.json"), "w") as f:
            json.dump(config, f)
    if not exists(join(results_folder, "config.json")):
        with open(join(results_folder, "config.json"), "w") as f:
            json.dump(config, f)



    train_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Instantiate model and pytorch lightning experiment
    model = Residual_AMR_Classifier(config)
    experiment = Classifier_Experiment(config, model)

    # Save summary of the model architecture
    if not exists(join(experiment_folder, "architecture.txt")):
        with open(join(experiment_folder, "architecture.txt"), "w") as f:
            f.write(model.__repr__())
    if not exists(join(results_folder, "architecture.txt")):
        with open(join(results_folder, "architecture.txt"), "w") as f:
            f.write(model.__repr__())


    # Setup training callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(output_folder, "checkpoints"),
                                          monitor="val_loss", filename="gst-{epoch:02d}-{val_loss:.4f}")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience
    )
    callbacks = [checkpoint_callback, early_stopping_callback]

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=join(output_folder, "logs/"))

    # Train model
    print("Training..")
    trainer = pl.Trainer(devices="auto", accelerator="auto", 
        default_root_dir=output_folder, max_epochs=args.n_epochs,#, callbacks=callbacks,
                        #  logger=tb_logger, log_every_n_steps=3
                        # limit_train_batches=6, limit_val_batches=4, limit_test_batches=4
                         )
    trainer.fit(experiment, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    
    print("Training complete!")

    # Test model
    print("Testing..")
    test_results = trainer.test(ckpt_path="best", dataloaders=test_loader)
    with open(join(results_folder, "test_metrics_{}.json".format(seed)), "w") as f:
        json.dump(test_results[0], f, indent=2)
        
    background_loader = DataLoader(test_dset, batch_size=100, shuffle=True)
    background = next(iter(background_loader))
    X_background = torch.cat(background[:3], dim=1)

    
    test_fi_loader = DataLoader(test_dset, batch_size=len(test_dset), shuffle=False)
    fi_batch = next(iter(test_fi_loader))
    X_fi_batch = torch.cat(fi_batch[:3], dim=1)
    
    experiment.model.eval()
    test_df["Predictions"] = experiment.test_predictions
    test_df.to_csv(join(results_folder, f"test_set_seed{seed}.csv"), index=False)
    
    if args.eval_importance:
        print("Evaluating feature importance")
        class ShapWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, X):
                species_idx = X[:, 0:1]
                x_spectrum = X[:, 1:6001]
                dr_tensor = X[:, 6001:]
                response = []
                dataset = []
                year = []
                batch = [species_idx, x_spectrum, dr_tensor, response, dataset, year]
                return experiment.model(batch)
            
        shap_wrapper = ShapWrapper(experiment.model)
        explainer = shap.DeepExplainer(shap_wrapper, X_background)
        shap_values = explainer.shap_values(X_fi_batch)
        column_names = ["Species"] + [f"Spectrum_{i}" for i in range(6000)] + [f"Fprint_{i}" for i in range(1024)]
        np.save(join(results_folder, f"shap_values_seed{seed}.npy"), shap_values)
        if not exists(join(output_folder, "shap_values_columns.json")):
            with open(join(output_folder, "shap_values_columns.json"), "w") as f:
                json.dump(column_names, f, indent=2)
        
        shap_values_df = pd.DataFrame(shap_values, columns=column_names)
        shap_values_df.to_csv(join(output_folder, "shap_values.csv"), index=False)
  
    
    print("Testing complete")





if __name__=="__main__":

    parser = ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="GNN")
    parser.add_argument("--experiment_group", type=str, default="ResAMR")
    parser.add_argument("--output_folder", type=str, default="outputs")
    parser.add_argument("--split_type", type=str, default="random", choices=["random", "drug_species_zero_shot", "drugs_zero_shot", "stratify_species"])


    parser.add_argument("--training_setup", type=int)
    parser.add_argument("--eval_importance", action="store_true")
    
    parser.add_argument("--seed", type=int, default=0)
    #EDIT: Expects list now
    parser.add_argument("--driams_dataset", nargs='+', type=str, default=["A","B","C","D"],
                    help="Specify one or more datasets from choices: A, B, C, D")
    parser.add_argument("--driams_long_table", type=str,
                        default="../processed_data/DRIAMS_combined_long_table.csv")
    parser.add_argument("--spectra_matrix", type=str,
                        default="../data/DRIAMS-B/spectra_binned_6000_2018.npy")
    parser.add_argument("--drugs_df", type=str,
                        # default="../processed_data/GNN_embeddings.csv")
                        default="../processed_data/drug_fingerprints.csv")

    parser.add_argument("--conv_out_size", type=int, default=512)
    parser.add_argument("--sample_embedding_dim", type=int, default=512)
    parser.add_argument("--drug_embedding_dim", type=int, default=512)
    

    parser.add_argument("--drug_emb_type", type=str, default="gnn_embedding", choices=["fingerprint", "vae_embedding", "gnn_embedding"])
    parser.add_argument("--fingerprint_class", type=str, default="morgan_1024", choices=["all", "MACCS", "morgan_512", "morgan_1024", "pubchem", "none"])
    parser.add_argument("--fingerprint_size", type=int, default=128)

    

    parser.add_argument("--n_hidden_layers", type=int, default=5)


    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()
    args.num_workers = os.cpu_count()


    if args.training_setup is not None:
        dataset, split_type, seed = TRAINING_SETUPS[args.training_setup]
        
        args.seed = seed
        args.driams_dataset = dataset
        args.split_type = split_type
    args.species_embedding_dim = 0
    
    args.experiment_name = args.experiment_name + f"_DRIAMS-{args.driams_dataset}_{args.split_type}"


    main(args)
