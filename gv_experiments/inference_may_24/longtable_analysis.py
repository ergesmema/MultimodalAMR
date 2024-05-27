import pandas as pd

def analyze_dataset(file_path, input_dataset):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Filter the DataFrame based on the input dataset
    filtered_df = df[df['dataset'] == input_dataset]
    
    # Calculate the number of unique species
    unique_species = filtered_df['species'].nunique()
    
    # Calculate the number of unique drugs
    unique_drugs = filtered_df['drug'].nunique()
    
    # Calculate the number of unique sample_id
    unique_sample_ids = filtered_df['sample_id'].nunique()
    
    # Calculate the top 10 most occurring species with unique sample_id and their occurrence number
    top_species = (
        filtered_df.drop_duplicates('sample_id')['species']
        .value_counts()
        .head(10)
    )
    
    # Calculate the top 10 drugs that have a response 0
    top_drugs_response_0 = (
        filtered_df[filtered_df['response'] == 0]['drug']
        .value_counts()
        .head(10)
    )
    
    # Calculate the top 10 drugs that have a response 1
    top_drugs_response_1 = (
        filtered_df[filtered_df['response'] == 1]['drug']
        .value_counts()
        .head(10)
    )
    
    return {
        'unique_species': unique_species,
        'unique_drugs': unique_drugs,
        'unique_sample_ids': unique_sample_ids,
        'top_species': top_species,
        'top_drugs_response_0': top_drugs_response_0,
        'top_drugs_response_1': top_drugs_response_1
    }

# Example usage
file_path = '/Users/em/Desktop/Uni-Spring24/XAIML-gitlab/b4-interpretable-antimicrobial-recommendation/MultimodalAMR/processed_data/DRIAMS_combined_long_table.csv'
input_dataset = 'A'
result = analyze_dataset(file_path, input_dataset)
print(result)
print(f'Number of unique species in dataset {input_dataset}: {result["unique_species"]}')
print(f'Number of unique drugs in dataset {input_dataset}: {result["unique_drugs"]}')
print(f'Number of unique sample IDs in dataset {input_dataset}: {result["unique_sample_ids"]}')
print('Top 10 most occurring species with unique sample IDs:')
print(result['top_species'])
print('Top 10 drugs that have a response 0:')
print(result['top_drugs_response_0'])
print('Top 10 drugs that have a response 1:')
print(result['top_drugs_response_1'])
