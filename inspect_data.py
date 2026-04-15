import pandas as pd
import io
import urllib.request

url = "https://raw.githubusercontent.com/aniketsharma00411/mba_placement_prediction/main/Placement_Data_Full_Class.csv"

# Download and load data
req = urllib.request.urlopen(url)
df = pd.read_csv(req)

print("=== FIRST 5 ROWS ===")
print(df.head())
print("\\n=== DATA TYPES AND MISSING VALUES ===")
info_df = pd.DataFrame({
    'Data Type': df.dtypes,
    'Missing Values': df.isnull().sum(),
    'Missing %': (df.isnull().sum() / len(df)) * 100,
    'Unique Values': df.nunique()
})
print(info_df)

print("\\n=== TARGET DISTRIBUTION (status) ===")
if 'status' in df.columns:
    print(df['status'].value_counts())
    print("\\nTarget (%)")
    print(df['status'].value_counts(normalize=True)*100)

if 'salary' in df.columns:
    print("\\n=== SALARY DISTRIBUTION ===")
    print(f"Total missing in salary: {df['salary'].isnull().sum()}")
    print("Salary summary for Placed students:")
    print(df[df['status'] == 'Placed']['salary'].describe())
