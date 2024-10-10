import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("inputs/sample_customer_data_for_exam.csv")

# Group the data by 'education' and calculate the average 'purchase_amount'
average_purchase_by_education = data.groupby('education')['purchase_amount'].mean().sort_values(ascending=False)

# Display the result
print("Average Purchase Amount by Education Level (sorted):")
print(average_purchase_by_education)

# Group the data by 'loyalty_status' and calculate the average 'satisfaction_score'
average_satisfaction_by_loyalty = data.groupby('loyalty_status')['satisfaction_score'].mean().sort_values(ascending=False)

# Display the result
print("Average Satisfaction Score by Loyalty Status (sorted):")
print(average_satisfaction_by_loyalty)

# Create a bar plot to compare 'purchase_frequency' across different 'region' values
plt.figure(figsize=(10, 6))
sns.barplot(x='region', y='purchase_frequency', data=data, palette='viridis')

# Add a title and labels
plt.title('Average Purchase Frequency by Region')
plt.xlabel('Region')
plt.ylabel('Average Purchase Frequency')

# Show the plot
plt.xticks(rotation=45)  # Rotate region labels for better readability if needed
plt.tight_layout()
plt.show()

# Calculate the total number of customers
total_customers = data.shape[0]

# Calculate the number of customers who used promotional offers (promotion_usage = 1)
customers_using_promotion = data[data['promotion_usage'] == 1].shape[0]

# Calculate the percentage of customers who used promotional offers
percentage_using_promotion = (customers_using_promotion / total_customers) * 100

# Display the result
print(f"Percentage of customers who used promotional offers: {percentage_using_promotion:.2f}%")

# Calculate the correlation coefficient between 'income' and 'purchase_amount'
correlation = data['income'].corr(data['purchase_amount'])

# Display the correlation coefficient
print(f"Correlation coefficient between income and purchase amount: {correlation:.2f}")

# Create a scatter plot to visualize the relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(x='income', y='purchase_amount', data=data, alpha=0.7)

# Add a title and labels
plt.title('Scatter Plot of Income vs. Purchase Amount')
plt.xlabel('Income')
plt.ylabel('Purchase Amount')

# Show the plot
plt.tight_layout()
plt.show()
