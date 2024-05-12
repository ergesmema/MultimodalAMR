import sys
# Assuming the following imports are correctly pointing to your project structure
sys.path.insert(0, "..")
sys.path.insert(0, "../../data_split")
import torch
from pytorch_lightning import Trainer
from argparse import Namespace
import json
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np

from multimodal_amr_tests_erges.models.classifier import Residual_AMR_Classifier
from multimodal_amr_tests_erges.models.data_loaders import DrugResistanceDataset_Fingerprints, DrugResistanceDataset_Embeddings
from multimodal_amr_tests_erges.experiments.pl_experiment import Classifier_Experiment


config_file = "/Users/em/Desktop/Uni-Spring24/XAIML-gitlab/b4-interpretable-antimicrobial-recommendation/MultimodalAMR/tests_erges/outputs/ResMLP/myExperiment1_DRIAMS-['A', 'B', 'C', 'D']_stratify_species/config.json"
checkpoint_file = "/Users/em/Desktop/Uni-Spring24/XAIML-gitlab/b4-interpretable-antimicrobial-recommendation/MultimodalAMR/tests_erges/outputs/ResMLP/myExperiment1_DRIAMS-['A', 'B', 'C', 'D']_stratify_species/0/lightning_logs/version_0/checkpoints/epoch=1-step=2770.ckpt"
sample_file = "/Users/em/Desktop/Uni-Spring24/XAIML/MultimodalAMR/DRIAMS-B/binned_6000/2018/ca568529-351a-43af-8cec-7175488f66ea.txt"
test_file = '/Users/em/Desktop/Uni-Spring24/XAIML-gitlab/b4-interpretable-antimicrobial-recommendation/MultimodalAMR/tests_erges/data/test_set_stratified_species.csv'
drugs_file = '/Users/em/Desktop/Uni-Spring24/XAIML-gitlab/b4-interpretable-antimicrobial-recommendation/MultimodalAMR/processed_data/drug_fingerprints.csv'


def load_model_from_checkpoint(experiment, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    new_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    experiment.model.load_state_dict(new_state_dict)
    return experiment

def load_model_for_inference(checkpoint_path, config):
    # config = Namespace(**config)
    model = Residual_AMR_Classifier(config)
    experiment = Classifier_Experiment(config, model)
    experiment = load_model_from_checkpoint(experiment, checkpoint_path)
    # model = load_model_from_checkpoint(experiment, checkpoint_path)
    experiment.model.eval()  # Set the model to evaluation mode
    return experiment

def inference(experiment, data_loader):
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            outputs = experiment.model(batch)  # These are logits if using BCEWithLogitsLoss
            probs = torch.sigmoid(outputs)  # Convert logits to probabilities
            print(probs)
            predicted_classes = (probs >= 0.5).int()  # Applying a threshold to get binary class predictions
            predictions.extend(predicted_classes.cpu().numpy()) 
    return predictions

if __name__ == '__main__':
    with open(config_file, 'r') as file:
        config = json.load(file)


    sample_data = pd.read_csv(sample_file, sep=" ", index_col=0)
    test_df = pd.read_csv(test_file)
    drugs_df = pd.read_csv(drugs_file, index_col=0)
    samples_list = sorted(test_df['sample_id'].unique())  # Ensure this matches your sample indexing strategy

    spectra_matrix = [sample_data["binned_intensity"].values]
    spectra_matrix = np.vstack(spectra_matrix)

    if config['drug_emb_type']=="fingerprint":
        test_dset = DrugResistanceDataset_Fingerprints(test_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
    elif config['drug_emb_type']=="vae_embedding" or config['drug_emb_type']=="gnn_embedding":
        test_dset = DrugResistanceDataset_Embeddings(test_df, spectra_matrix, drugs_df, samples_list)

    test_loader = DataLoader(test_dset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    experiment = load_model_for_inference(checkpoint_file, config)
    predictions = inference(experiment, test_loader)

    print(predictions)
    # print(experiment.test_predictions)




