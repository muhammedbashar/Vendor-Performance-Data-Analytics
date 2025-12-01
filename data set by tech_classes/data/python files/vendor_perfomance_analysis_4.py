import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sqlalchemy import create_engine
from scipy.stats import ttest_ind 
import scipy.stats as stats
warnings.filterwarnings('ignore')


# ----- LOADING DATASET -----
# creating database connection
engine_url = f"mysql+pymysql://{'root'}:{'8520147'}@{'localhost'}:{3306}/{'tech_classes'}"
engine = create_engine(engine_url)

# fetching vendor summary data
df = pd.read_sql("SELECT * FROM vendor_sales_summary",con=engine)
#print(df.head())

""""
## ===== EXPLORATORY DATA ANALYSIS ===== ##
   -> previously, we examinated the various tables in the database to identify key variables, understand their relationships, 
        and determine which ones should be included in the final analysis.
   -> in this phase of EDA, we will analyse the resultant table to gain insights into the distribution of each column, This will help us understand data patterns, 
       identify anomalies, and ensure data quality before proceeding with further analysis."""


# SUMMARY STATISTICS 
print(df.describe().T)

# Distribution plots for NUmerical Columns
numerical_columns = df.select_dtypes(include=np.number).columns
print(numerical_columns)

plt.figure(figsize=(15,10))
for i, col in enumerate(numerical_columns):
    plt.subplot(4, 4, i+1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)

plt.tight_layout()
plt.show()


# outlier detection with boxplots
plt.figure(figsize=(15,10))
for i, col in enumerate(numerical_columns):
    plt.subplot(4,4,i+1)
    sns.boxenplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

""" 
SUMMARY STATISTICS INSIGHTS :- 
    >> NEGATIVE & ZERO VALUES :-
        -> Gross profit : minimum value is -52,002.78, indicating losses, some products or transactions may be selling at a loss due to high costs or selling at discouts lower than the purchase price.
        -> Profit margin : has a minimum of -infinite, which suggests cases where revenue is zero or even lower than costs.
        -> Total Sales Quantity & Sales Dollars : minimum values are 0, meaning some products were purchased but never sold. These could be slow-moving or obsolete stock.

    >> OUTLIERS INDICATED BY HIGH STANDARD DEVIATIONS:
        -> Purchase & Actual Prices : The max values (5,681.81 & 7,499.99) are significantly higher than the mean (24.39 & 35.64), indicating potential premium products.
        -> Freight cost : huge variation, from 0.09 to 257,032.07, suggests logistics inefficiencies or bulk shipments.
        -> Stock Turnover : ranges form 0 to 274.5, implying some products sell extremely fast while others remain in stock indefinitely, 
            value more than 1 indicates that sold quantity for that product is higher than purchased quantity due to either sales are being fulfilled from older stock. 
"""


# lets filter the data by removing inconsistencies
df = pd.read_sql("""
                 SELECT * 
                 FROM vendor_sales_summary
                 WHERE gross_profit > 0
                 AND profit_margin > 0
                 AND total_sales_quantity > 0 """, con=engine)

print(df)
print()

plt.figure(figsize=(15,10))
for i, col in enumerate(numerical_columns):
    plt.subplot(4, 4, i+1)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)

plt.tight_layout()
plt.show()


print(f"our df columns names are :\n {df.columns}")

# count plots for categorical columns
categorical_cols = ['VendorName', 'Description']

plt.figure(figsize=(12,5))
for i , col in enumerate(categorical_cols):
    plt.subplot(1, 2, i+1)
    sns.countplot(y=df[col], order=df[col].value_counts().index[:10]) # top 10 categories
    plt.title(f"count plot of {col}")
plt.tight_layout()
plt.show()


# CORRELATION HEATMAP
plt.figure(figsize=(12,8))
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

""" 
CORRELATION INSIGHTS :-
    -> 'purchase_price' has weak correlation with 'total_sales_dollars'(-0.012) and 'gross_profit'(-0.016), 
            suggesting that price variation do not significantly impact sales revenue or profit.
    -> strong correlation between 'total_purchase_quantity' and 'total_sales_quantity'(0.999), confirming efficient inventory turnover.
    -> Negative correlation between 'profit_margin' & 'total_sales_price' (-0.179) suggests that as sales price increases, margines decreases, possibly due to competitive pricing pressures.
    -> 'stock_turn_over' has weak negative correlation with both 'gross_profit' (-0.038) and 'profit_margin' (-0.055), indicating that faster turnover does not necessarily result in higher profitablity.

"""



# ========== DATA ANALYSIS ========== #
''' identify brands that needs promotional or pricing adjustments which exhibit lower perfomance but higher profit margins '''
brand_perfomance = df.groupby('Description').agg({
                         'total_sales_dollars':'sum',
                           'profit_margin':'mean'}).reset_index()



low_margin_sales_threshold = brand_perfomance['total_sales_dollars'].quantile(0.15)
high_margin_sales_threshold = brand_perfomance['profit_margin'].quantile(0.85)

print('low_margin_sales_thresold :',low_margin_sales_threshold)
print()
print('high_margin_sales_thresold :',high_margin_sales_threshold)


# Filter brands with low sales but high profit margines
target_brands = brand_perfomance[
    (brand_perfomance['total_sales_dollars'] <= low_margin_sales_threshold) &
    (brand_perfomance['profit_margin'] >= high_margin_sales_threshold)
]

print("Brands with low sales but high profit margins:")
print(target_brands.sort_values('total_sales_dollars'))


brand_perfomance = brand_perfomance[brand_perfomance['total_sales_dollars'] < 10000] # for better visualisation

plt.figure(figsize=(10,6))

# Scatter plots
sns.scatterplot(data=brand_perfomance, x='total_sales_dollars', y='profit_margin',
                color='green', label='All brands', alpha=0.2)

sns.scatterplot(data=target_brands, x='total_sales_dollars', y='profit_margin',
                color='red', label='Target brands')

# Threshold lines
plt.axhline(high_margin_sales_threshold, linestyle='--', color='black', label='High margin threshold')

plt.axvline(low_margin_sales_threshold, linestyle='--', color='black', label='Low sales threshold')   # FIXED "label" typo

# Labels & title
plt.xlabel('Total Sales ($)')
plt.ylabel('Profit Margin ($)')  # FIXED "ylabel" typo

plt.title('Brands for Promotional or Pricing Adjustments')
plt.legend()
plt.grid(True)
plt.show()

''' which vendors and brands demonstrates the highest sales perfomance ? '''
# top vendors & brands by sales perfomance
top_vendors = df.groupby('VendorName')['total_sales_dollars'].sum().nlargest(10)
top_brands = df.groupby('Description')['total_sales_dollars'].sum().nlargest(10)

print(f"top vendors are : {top_vendors}\n")
print(f"top brands are : {top_brands} \n")

def format_dollars(value): # this function for minimalising the huge number value to "M" or "K"
    if value >= 1000000:
        return f"{value / 1000000:.2f}M"
    elif value >= 1000:
        return f"{value / 1000:.2f}K"
    else:
        return str(value) 
    

print(top_brands.apply(format_dollars))
print(top_vendors.apply(format_dollars))


plt.figure(figsize=(15,5))

# ====== TOP VENDORS ======
plt.subplot(1, 2, 1)
ax1 = sns.barplot(y=top_vendors.index, x=top_vendors.values, palette='Blues_r')
plt.title("Top 10 Vendors by Sales")

# Add labels on bars
for bar in ax1.patches:
    ax1.text(
        bar.get_width() + (bar.get_width() * 0.02),
        bar.get_y() + bar.get_height() / 2,
        format_dollars(bar.get_width()),
        ha='left',
        va='center',
        fontsize=10,
        color='black'
    )

# ====== TOP BRANDS ======
plt.subplot(1, 2, 2)
ax2 = sns.barplot(y=top_brands.index.astype(str), x=top_brands.values, palette='Reds_r')
plt.title("Top 10 Brands by Sales")

# Add labels on bars
for bar in ax2.patches:
    ax2.text(
        bar.get_width() + (bar.get_width() * 0.02),
        bar.get_y() + bar.get_height() / 2,
        format_dollars(bar.get_width()),
        ha='left',
        va='center',
        fontsize=10,
        color='black'
    )

plt.tight_layout()
plt.show()



''' which vendors contribute the most to total purchase dollars ?'''
vendor_perfomance = df.groupby('VendorName').agg({
                    'total_purchase_dollars':'sum',
                    'gross_profit':'sum',
                    'total_sales_dollars':'sum'
                }).reset_index()


vendor_perfomance['purchase_contribution%'] = (
    vendor_perfomance['total_purchase_dollars'] 
    / vendor_perfomance['total_purchase_dollars'].sum() * 100
)

vendor_perfomance = round(
    vendor_perfomance.sort_values('purchase_contribution%', ascending=False), 2
)

# Top 10 Vendors
top_vendors = vendor_perfomance.head(10)

top_vendors['total_sales_dollars'] = top_vendors['total_sales_dollars'].apply(format_dollars)
top_vendors['total_purchase_dollars'] = top_vendors['total_purchase_dollars'].apply(format_dollars)
top_vendors['gross_profit'] = top_vendors['gross_profit'].apply(format_dollars)

print(top_vendors)
print()
print(top_vendors['purchase_contribution%'].sum()) # here we get know that all those top 10 vendors hold 66% of total puchase contribution from WHOLE contributions

top_vendors['cumulative_contribution%'] = top_vendors['purchase_contribution%'].cumsum()  # add a cumulative value colums to top_vendors df
print(top_vendors)


# =================== PLOTTING =====================
fig, ax1 = plt.subplots(figsize=(10,6))

# Bar plot (Purchase Contribution)
sns.barplot(
    x=top_vendors['VendorName'], 
    y=top_vendors['purchase_contribution%'], 
    palette='mako', ax=ax1
)

# Add labels on bars
for i, value in enumerate(top_vendors['purchase_contribution%']):
    ax1.text(i, value - 1, f"{value:.2f}%", ha='center', fontsize=10, color='white')

# Line plot for cumulative %
ax2 = ax1.twinx()
ax2.plot(
    top_vendors['VendorName'], 
    top_vendors['cumulative_contribution%'], 
    color='red', marker='o', linestyle='dashed', label='Cumulative %'
)

# X-axis vertical labels
ax1.set_xticklabels(top_vendors['VendorName'], rotation=90, ha='center')

ax1.set_ylabel('Purchase Contribution (%)', color='blue')
ax2.set_ylabel('Cumulative Contribution (%)', color='red')
ax1.set_xlabel('Vendors')
ax1.set_title('Pareto Chart: Vendor Contributions to Total Purchase')

ax2.axhline(y=100, color='gray', linestyle='dashed', alpha=0.7)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()



''' how much of total procurement is dependent on the top vendors? '''
print(f"total puchased contribution of top 10 vendors is {round(top_vendors['purchase_contribution%'].sum(),2)}%")

# =================== PLOTTING =====================
vendors = list(top_vendors['VendorName'].values)
purchase_contributions = list(top_vendors['purchase_contribution%'].values)

total_contribution = sum(purchase_contributions)
remaining_contribution = 100 - total_contribution

# Add "other vendors"
vendors.append('other vendors')
purchase_contributions.append(remaining_contribution)

# Donut chart
fig, ax = plt.subplots(figsize=(8,8))
wedges, texts, autotexts = ax.pie(
    purchase_contributions,
    labels=vendors,
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.85,
    colors=plt.cm.Paired.colors
)

# Draw white center circle for donut effect
center_circle = plt.Circle((0,0), 0.70, fc='white')
plt.gca().add_artist(center_circle)

# Add total contribution annotation
plt.text(
    0, 0,
    f"Top 10 total:\n{total_contribution:.1f}%",
    fontsize=14,
    fontweight='bold',
    ha='center',
    va='center'
)

plt.title("Top 10 Vendors' Purchase Contribution (%)")
plt.tight_layout()
plt.show()




''' Does purchasing in bulk reduce the unit price, and what is the optional purchase volume for cost savings? '''
df['unit_purchase_price'] = df['total_purchase_dollars'] / df['total_purchase_quantity']  # created a new columns
print("unit purchased price is :\n",df['unit_purchase_price'])

df['order_size'] = pd.qcut(df['total_purchase_quantity'], q=3, labels=['small','medium','large'])
print(df[['order_size','total_purchase_quantity']])

print(df.groupby('order_size')[['unit_purchase_price']].mean())

# =================== PLOTTING =====================

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='order_size', y='unit_purchase_price', palette='Set2')
plt.title('IMPACT OF BULK PURCHASING ON THIS PRICE')
plt.xlabel('Order Size')
plt.ylabel('Average unit purchase price')
plt.show()

""" 
            KEY THOUGHTS FROM LAST BOXPLOT 
-> vendors buying in bulk (large order size) get the lowest unit price($ 10.78 per unit), meaning higher margins if they can manage inventory efficiency.
-> the price difference between small and large orders is substantial (-72% reduction in unit cost) 
-> this suggests that bulk pricing strategies successfully encourage vendors to purchase in larger volumes, leading to higher overall sales despite lower per-unit revenue.

"""


''' which vendors have low inventory turnover, indicating excess stock and slow-moving products? '''
print(f"df columns are :\n {df.columns}\n")
print(df[df['stock_turn_over'] < 1].groupby('VendorName')[['stock_turn_over']].mean().sort_values('stock_turn_over', ascending=True).head(10))



''' how much capital is locked in unsold inventory per vendor, and which vendors contribute the most to it? '''

print(f"df columns are :\n {df.columns}\n")
df['unsold_inventory_value'] = (df['total_purchase_quantity'] - df['total_sales_quantity']) * df['PurchasePrice']
print('total unsold captial :', format_dollars(df['unsold_inventory_value'].sum()))

# Aggregate capital locked per vendor
inventory_value_per_vendor = df.groupby('VendorName')['unsold_inventory_value'].sum().reset_index()

# Sort Vendors with the Highest locked capital:
inventory_value_per_vendor = inventory_value_per_vendor.sort_values(by='unsold_inventory_value', ascending=False)
inventory_value_per_vendor['unsold_inventory_value'] = inventory_value_per_vendor['unsold_inventory_value'].apply(format_dollars)
print(inventory_value_per_vendor.head(10))
print()



''' what is the 95% confidence intervals for profit margins of top-performing and low-performing vendors. '''
# ---- TOP & LOW VENDORS ----

top_threshold = df['total_sales_dollars'].quantile(0.75)
low_threshold = df['total_sales_dollars'].quantile(0.25)

top_performing_vendors = df[df['total_sales_dollars'] >= top_threshold]['profit_margin'].dropna()
low_performing_vendors = df[df['total_sales_dollars'] <= low_threshold]['profit_margin'].dropna()


# ---- CONFIDENCE INTERVAL FUNCTION ----
def confidence_interval(data, confidence=0.95):
    mean = np.mean(data)   # average value
    se = np.std(data, ddof=1) / np.sqrt(len(data)) #std error (value between highest and mean is equal with value between lowest and mean)
    t_val = stats.t.ppf((1 + confidence) / 2, df=len(data) - 1)
    moe = t_val * se
    return {
        "mean": mean,
        "lower_bound": mean - moe,
        "upper_bound": mean + moe,
        "margin_of_error": moe
    }


# ---- GET INTERVALS ----
top_ci = confidence_interval(top_performing_vendors)  # ci (confidence intervals)
low_ci = confidence_interval(low_performing_vendors)

top_mean, top_lower, top_upper = top_ci["mean"], top_ci["lower_bound"], top_ci["upper_bound"]
low_mean, low_lower, low_upper = low_ci["mean"], low_ci["lower_bound"], low_ci["upper_bound"]


print(f"Top vendors 95% CI: ({top_lower:.2f}, {top_upper:.2f}), Mean: {top_mean:.2f}")
print(f"Low vendors 95% CI: ({low_lower:.2f}, {low_upper:.2f}), Mean: {low_mean:.2f}")


# ---- PLOTTING ----

plt.figure(figsize=(12,6))

# Top vendors
sns.histplot(top_performing_vendors, kde=True, color='blue', bins=30, alpha=0.5, label="Top vendors")
plt.axvline(top_lower, color='blue', linestyle='--', label=f"Top lower: {top_lower:.2f}")
plt.axvline(top_upper, color='blue', linestyle='--', label=f"Top upper: {top_upper:.2f}")
plt.axvline(top_mean,  color='blue', linestyle='-',  label=f"Top mean: {top_mean:.2f}")

# Low vendors
sns.histplot(low_performing_vendors, kde=True, color='red', bins=30, alpha=0.5, label="Low vendors")
plt.axvline(low_lower, color='red', linestyle='--', label=f"Low lower: {low_lower:.2f}")
plt.axvline(low_upper, color='red', linestyle='--', label=f"Low upper: {low_upper:.2f}")
plt.axvline(low_mean,  color='red', linestyle='-',  label=f"Low mean: {low_mean:.2f}")


plt.title("Confidence Interval Comparison: Top vs Low Vendors (Profit Margin)")
plt.xlabel("Profit Margin (%)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()    

"""
            KEY THOUGHT FROM THE CHART
- > the confidence interval for low-performing vendors (40.48% to 42.62%) is significantly higher than that of top-performing vendors (30.74% to 31.61%).
- > this suggests that vendors with lower sales tend to maintain higher profit margins, potentially due to premium pricing or lower operational costs.
- > for high-performing vendors: if they aim to improve profitablity, they could explore selecctive price adjustments, cost optimisation, or bundling strategies.
- > for low-performing vendors: despite higher margins, their low sales volume might indicate a need for better marketing, competitive pricing, or improved distribution strategies.

"""


'''' is there a significant difference in profit margins between top-performing vendors? '''
''' 
Hypothesis :
    H0 (Null Hypothesis): there is no significant difference in the mean profit margins of top-performing and low-performing vendors.
    H1 (Alternative Hypothesis): the mean profit margins of top-performing and low-performing vendors are significantly different.
    
'''
top_threshold = df['total_sales_dollars'].quantile(0.75)
low_threshold = df['total_sales_dollars'].quantile(0.25)

top_performing_vendors = df[df['total_sales_dollars'] >= top_threshold]['profit_margin'].dropna()
low_performing_vendors = df[df['total_sales_dollars'] <= low_threshold]['profit_margin'].dropna()

# perform two-sample T-test
t_stat, p_value = ttest_ind(top_performing_vendors, low_performing_vendors, equal_var=False)

# print results
print(f"T-statistics: {t_stat:.4f}, P-value: {p_value:.4f}")
if p_value < 0.05:
    print('Reject H0 : There is a significant difference in profit margins between top and low-performing vendors.')
else:
    print('Fail to Reject H0 : No significat difference in profit margins.')



