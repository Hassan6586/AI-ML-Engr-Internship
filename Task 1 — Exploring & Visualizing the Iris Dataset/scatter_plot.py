import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------
# LOAD IRIS DATASET
# ----------------------------------------
# seaborn built-in dataset
df = sns.load_dataset("iris")

# If you have iris.csv, use:
# df = pd.read_csv("iris.csv")

# ----------------------------------------
# BASIC INSPECTION
# ----------------------------------------
print("\n=== Dataset Shape ===")
print(df.shape)

print("\n=== Column Names ===")
print(df.columns.tolist())

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Dataset Info ===")
df.info()

print("\n=== Summary Statistics ===")
print(df.describe())


# ----------------------------------------
# SCATTER PLOT
# ----------------------------------------
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="species")
plt.title("Scatter Plot: Sepal Length vs Sepal Width")
plt.show()


# ----------------------------------------
# HISTOGRAMS
# ----------------------------------------
df.hist(figsize=(10,6), bins=12)
plt.suptitle("Histograms of Iris Features")
plt.show()


# ----------------------------------------
# BOXPLOT FOR OUTLIERS
# ----------------------------------------
plt.figure(figsize=(10,6))
sns.boxplot(data=df)
plt.title("Boxplots for Iris Dataset Features")
plt.show()
