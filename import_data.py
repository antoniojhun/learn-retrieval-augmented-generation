import pandas as pd

# Step 1: Read the CSV file
df = pd.read_csv('top_rated_wines.csv')

# Step 2: Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Step 3: Display the descriptive statistics of the DataFrame
print("\nDescriptive statistics of the DataFrame:")
print(df.describe())

# Step 4: Convert the DataFrame to a list of dictionaries
records = df.to_dict('records')

# Step 5: Display the first few records to verify the conversion
print("\nFirst few records as dictionaries:")
print(records[:5])
