import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Penguins Data.csv")

print("First 10 rows of the dataset:")
print(df.head(10))


print("\nCheck if any null values exist:")
print(df.isnull().values.any())


print("\nTotal null values in each column:")
print(df.isnull().sum())


subset = df.iloc[:500, :]   

plt.figure(figsize=(12,8))
sns.heatmap(subset.isnull(), cbar=False, cmap='viridis')
plt.title("Visualization of Null Values")
plt.show()


categorical_columns = df.select_dtypes(include=['object']).columns

print("\nCategorical columns:", categorical_columns)

for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nCategorical null values handled using mode.")


numerical_columns = df.select_dtypes(include=['int64','float64']).columns

print("\nNumerical columns:", numerical_columns)


for col in numerical_columns:
    df[col].fillna(df[col].mean(), inplace=True)

print("\nNumerical null values handled using mean.")


print("\nNull values after handling:")
print(df.isnull().sum())


df_cleaned = df.dropna()

print("\nFinal cleaned dataset preview:")
print(df_cleaned.head(10))