import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("inputs/sample_customer_data_for_exam.csv")

# Scatter plot of purchase frequency vs purchase amount, color-coded by loyalty status
plt.figure(figsize=(10, 6))
sns.scatterplot(x='purchase_frequency', y='purchase_amount', hue='loyalty_status', data=data, alpha=0.7, palette='deep')

# Add title and labels
plt.title('Purchase Frequency vs. Purchase Amount by Loyalty Status')
plt.xlabel('Purchase Frequency')
plt.ylabel('Purchase Amount')

# Show the plot
plt.legend(title='Loyalty Status')
plt.tight_layout()
plt.show()

# Calculate average purchase amount based on promotion usage
average_purchase_by_promotion = data.groupby('promotion_usage')['purchase_amount'].mean().reset_index()

# Display the results
print("\nAverage Purchase Amount for Customers Who Used Promotions vs Those Who Didn’t:")
print(average_purchase_by_promotion)

# Create a violin plot for satisfaction score by loyalty status
plt.figure(figsize=(10, 6))
sns.violinplot(x='loyalty_status', y='satisfaction_score', data=data, palette='muted')

# Add title and labels
plt.title('Distribution of Satisfaction Score by Loyalty Status')
plt.xlabel('Loyalty Status')
plt.ylabel('Satisfaction Score')

# Show the plot
plt.tight_layout()
plt.show()

# Calculate the count of promotion usage across product categories
promotion_counts = data.groupby(['product_category', 'promotion_usage']).size().unstack()

# Create a stacked bar chart
promotion_counts.plot(kind='bar', stacked=True, color=['lightblue', 'lightcoral'], figsize=(10, 6))

# Add title and labels
plt.title('Proportion of Promotion Usage Across Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Count')

# Show the plot
plt.legend(title='Promotion Usage', labels=['Did Not Use', 'Used'])
plt.tight_layout()
plt.show()

# Calculate the correlation coefficient between satisfaction score and purchase frequency
correlation = data['satisfaction_score'].corr(data['purchase_frequency'])

# Display the correlation coefficient
print(f"Correlation coefficient between satisfaction score and purchase frequency: {correlation:.2f}")
