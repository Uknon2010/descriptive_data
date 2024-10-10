import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("inputs/sample_customer_data_for_exam.csv")

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Generate summary statistics for numerical columns
summary_stats = data.describe()
print("\nSummary statistics for numerical columns:")
print(summary_stats)

# Select only numerical columns
numerical_data = data.select_dtypes(include=[np.number])

# Calculate the correlation matrix for numerical variables
correlation_matrix = numerical_data.corr()

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)

# Show the plot
plt.title("Correlation Heatmap of Numerical Variables")
plt.show()

# Create histograms for 'age' and 'income' columns
plt.figure(figsize=(14, 6))

# Plot histogram for 'age'
plt.subplot(1, 2, 1)
plt.hist(data['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Plot histogram for 'income'
plt.subplot(1, 2, 2)
plt.hist(data['income'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Income Distribution')
plt.xlabel('Income')
plt.ylabel('Frequency')

# Show the plots
plt.tight_layout()
plt.show()

# Create a box plot to show the distribution of 'purchase_amount' across 'product_category'
plt.figure(figsize=(10, 6))
sns.boxplot(x='product_category', y='purchase_amount', data=data, palette='Set2')

# Add a title and labels
plt.title('Distribution of Purchase Amount Across Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Purchase Amount')

# Show the plot
plt.xticks(rotation=45)  # Rotate category labels if necessary
plt.tight_layout()
plt.show()

# Count the number of customers in each gender category
gender_counts = data['gender'].value_counts()

# Create a pie chart
plt.figure(figsize=(7, 7))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightblue', 'lightgreen'], startangle=90, wedgeprops={'edgecolor': 'black'})

plt.title('Proportion of Customers by Gender')

# Show the plot
plt.show()

