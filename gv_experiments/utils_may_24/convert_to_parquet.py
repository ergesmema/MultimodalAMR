import pandas as pd
import os

# Specify the directory containing the files
directory = '/Users/em/Desktop/Uni-Spring24/XAIML/MultimodalAMR/DRIAMS-B/binned_6000/2018'

# Iterate over each file in the directory
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    
    # Check if the file is a file and not a directory
    if os.path.isfile(file_path):
        # Read the file with space as separator and the first column as index
        df = pd.read_csv(file_path, sep=' ', index_col=0)
        
        # Convert the DataFrame to Parquet, saving in the same directory with .parquet extension
        parquet_path = os.path.join("parquet_data/", os.path.splitext(filename)[0] + '.parquet')
        df.to_parquet(parquet_path, engine='pyarrow')

print("Conversion complete.")
