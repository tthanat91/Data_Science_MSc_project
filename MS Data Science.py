import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("/Users/THANAT/Documents/Research Projects/MSc project/IS/Data/RC.csv")
df = df.dropna(axis=1, how='all')  # Drop fully empty columns

# === Step 1: Missing Data Percentage ===
missing_percent = df.isnull().mean() * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=missing_percent.values, y=missing_percent.index, palette="Reds_r")
plt.xlabel("Missing Data (%)")
plt.title("Percentage of Missing Data by Feature")
plt.tight_layout()
plt.savefig("missing_data_percent.png")
plt.show()

# === Filter out features with >50% missing values (except exceptions) ===
exceptions_to_keep = ['postpone_neoadj', 'tnt']
features_to_drop = [col for col in missing_percent.index
                    if missing_percent[col] > 50 and col not in exceptions_to_keep]
df = df.drop(columns=features_to_drop)

print("Dropped features due to >50% missingness (except clinical exceptions):")
print(features_to_drop)

# === Step 2: Convert fields for EDA ===
binary_columns = ['neoadj', 'tnt', 'postpone_neoadj',
                  'lvi', 'curative_intend', 'recur']
df[binary_columns] = df[binary_columns].replace({'Yes': 1, 'No': 0}).infer_objects(copy=False)
df['sex'] = df['sex'].replace({'Male': 1, 'Female': 0}).infer_objects(copy=False)
df['cea_binary'] = df['cea_level'].apply(lambda x: 1 if x >= 5 else 0 if pd.notnull(x) else None)

# === Step 3: Correlation Matrix ===
numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=0.5)
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.show()

# === Step 4: Combined Histograms for Continuous Variables ===
num_features = ['age', 'no_ln']
num_plots = len(num_features)

# Adjust layout (1 row, 2 columns)
fig, axes = plt.subplots(1, num_plots, figsize=(12, 4))

for i, col in enumerate(num_features):
    if col in df.columns:
        sns.histplot(df[col].dropna(), kde=True, bins=20, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("combined_histograms.png")
plt.show()


# === Step 5: Combined Count Plots ===
# === Recode 'margin' to binary: 'Negative' vs. 'Positive'
df['margin_binary'] = df['margin'].apply(
    lambda x: 'Negative' if isinstance(x, str) and x.strip().lower() == 'negative' else 'Positive'
)
cat_features = ['sex', 'tumor_loc', 'tumor_grade', 'cea_binary',
                'neoadj', 'tnt', 'recur', 'margin_binary']

num_plots = len(cat_features)
cols = 3  # number of columns in the grid
rows = (num_plots + cols - 1) // cols  # automatically determine rows

fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
axes = axes.flatten()  # make it easier to index

for i, col in enumerate(cat_features):
    if col in df.columns:
        sns.countplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f"Count Plot: {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis='x', rotation=45)

# Hide unused subplots if total plots < grid size
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("combined_count_plots.png")
plt.show()


# === Step 6: Combined Boxplots Grouped by Recurrence ===
boxplot_features = ['age', 'no_ln']
num_plots = len(boxplot_features)

fig, axes = plt.subplots(1, num_plots, figsize=(12, 4))

for i, col in enumerate(boxplot_features):
    if col in df.columns:
        sns.boxplot(data=df, x='recur', y=col, ax=axes[i])
        axes[i].set_title(f"{col} by Recurrence")
        axes[i].set_xlabel("Recurrence (0=No, 1=Yes)")
        axes[i].set_ylabel(col)

plt.tight_layout()
plt.savefig("combined_boxplots_by_recur.png")
plt.show()
