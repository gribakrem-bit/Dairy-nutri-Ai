import pandas as pd
import matplotlib.pyplot as plt

# Load dairy dataset
df = pd.read_csv("data/dairy_products.csv")

# Display basic information
print("DairyNutri AI Dataset")
print("---------------------")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nNutritional statistics:")
print(df.describe())

# Average nutrients by category
category_means = df.groupby("Category")[
    ["Energy_kcal", "Fat_g", "Protein_g", "Carbohydrate_g"]
].mean()

print("\nAverage nutritional composition by category:")
print(category_means)

# Visualization
category_means.plot(kind="bar", figsize=(10, 6))

plt.title("Average Nutritional Composition of Dairy Products")
plt.xlabel("Dairy Category")
plt.ylabel("Average Amount")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("dairy_nutrition_comparison.png")
plt.show()
