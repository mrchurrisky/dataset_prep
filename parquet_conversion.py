import pandas as pd
import json

# Step 1: Load the parquet file
df = pd.read_parquet('your_file.parquet')

# Step 2: Inspect and preprocess
print(df.head())
df = df.dropna()
df_finetune = df[['input_text', 'output_text']]

# Step 3: Convert to JSONL format
def df_to_jsonl(dataframe, output_file):
    with open(output_file, 'w') as f:
        for _, row in dataframe.iterrows():
            json_record = {
                "prompt": row['input_text'],
                "completion": row['output_text']
            }
            f.write(json.dumps(json_record) + '\n')

# Write the data to a JSONL file
df_to_jsonl(df_finetune, 'finetune_data.jsonl')
