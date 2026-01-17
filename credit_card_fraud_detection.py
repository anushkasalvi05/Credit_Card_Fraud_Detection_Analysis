#%%
#!pip install statsmodels
!pip install scikit-learn


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind


# %%

"""
In this section, we define the path to the dataset. The **fraudTrain.csv** file contains simulated credit card transaction data, which we load into a Pandas DataFrame.
"""

Data_Path = "fraudTrain.csv"
df = pd.read_csv(Data_Path)



# %%
print("Shape:", df.shape)

# %%
print(df.head())


 # %%
''' The first few rows show that each transaction contains:
# - full timestamp of the purchase
# - merchant name with some prefixed by fraud_
# - transaction category
# - amount
# - customer demographic details (name, address, job, DOB, etc.)
# - location coordinates for both customer and merchant
# - the target variable 'is_fraud'

# This confirms that the dataset captures realistic transaction behavior.
# Some columns like 'first', 'last', 'street', 'dob', and job are not useful for prediction and will likely be removed later.
'''


# %%

"""
This section prints the **data types** of each column to check the format (e.g., numeric, categorical). This step is important for ensuring that the data is in the correct format for modeling.
"""

print("\nData types:")
print(df.dtypes)


# %%

"""
Here, we check for missing values in the dataset using **`isna().sum()`**, which counts how many null values are present in each column. Missing data handling is a crucial step in data preprocessing.
"""

print("\nMissing values per column:")
print(df.isna().sum())
missing_pct = df.isna().sum() / len(df) * 100
print("\nPercentage of missing values per column:")
print(missing_pct.sort_values(ascending=False))
'''All columns show 0 missing values.'''


 # %%
drop_cols = [
    "Unnamed: 0",  
    "cc_num",      
    "first",
    "last",
    "street",
    "job",
    "dob",
    "trans_num",   
    "unix_time",   
    "city_pop",    
    "zip"     ]     
df = df.drop(columns=drop_cols)
print("\nShape after dropping unnecessary columns:", df.shape)
print(df.head())

"""
Here, we define a list of columns to be dropped from the dataset. These columns are either irrelevant (e.g., personal information like name and job) or redundant (e.g., `Unnamed: 0` which is an index column).
We then drop these columns using **`drop()`** to focus the dataset on features relevant to fraud detection.
"""

 # %%
'''
After dropping these columns, the dataset is cleaner and more focused on
transaction-level behavior rather than personal identity details.
We still keep:
- trans_date_trans_time (time of transaction)
- merchant and category
- amt (transaction amount)
- location coordinates (lat/long and merch_lat/merch_long)
- gender, city, state
- is_fraud (target)
These are the features we want to use to explore fraud patterns and build models.
'''


# %%
"""
In this section, we convert the `trans_date_trans_time` column from a string format into a **datetime** object using **`pd.to_datetime()`**. This allows us to perform time-based analysis on transaction dates and times.
"""

df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])


# %%
"""
This part displays the distribution of the target variable `is_fraud`, showing how many transactions are legitimate and how many are fraudulent. This is crucial for understanding class imbalance in the dataset.
"""

print("\nFraud vs Legit counts:")
print(df["is_fraud"].value_counts())

print("\nFraud vs Legit proportions:")
print(df["is_fraud"].value_counts(normalize=True))

plt.figure(figsize=(5, 3))
sns.countplot(data=df, x="is_fraud")
plt.xticks([0, 1], ["Legit (0)", "Fraud (1)"])
plt.title("Class Distribution: Legit vs Fraud")
plt.tight_layout()
plt.show()


# %%
'''The class distribution plot shows that the dataset is extremely imbalanced. 
Out of 1.29 million total transactions, almost all are legitimate, while only a tiny fraction represent fraud. 
Specifically, fraud accounts for less than 1% of all observations. 
This imbalance is typical in real-world financial fraud settings because fraudulent transactions are rare compared to the large volume of normal activity.'''


# %%
# Merchant Level Fraud Analysis - SMART Question 1
merchant_stats = (
    df.groupby("merchant")["is_fraud"]
      .agg(fraud_rate="mean", num_fraud="sum", total_txn="count")
      .sort_values(by="fraud_rate", ascending=False)
)

print("\nTop 10 merchants by fraud rate:")
print(merchant_stats.head(10))


# %%
merchant_clean = merchant_stats[merchant_stats["total_txn"] >= 100]
print("\nMerchants with >100 transactions:", merchant_clean.shape[0])


# %%
plt.figure(figsize=(12,6))
top_merchants = merchant_clean.sort_values(by="fraud_rate", ascending=False).head(20)

sns.barplot(
    data=top_merchants,
    x=top_merchants.index,
    y="fraud_rate",
    palette="Blues_r"
)

plt.xticks(rotation=90)
plt.title("Top 20 Merchants by Fraud Rate")
plt.ylabel("Fraud Rate")
plt.xlabel("Merchant")
plt.tight_layout()
plt.show()


# %%
df["merchant_risk_score"] = df["merchant"].map(merchant_stats["fraud_rate"])
print(df[["merchant", "merchant_risk_score"]].head())


# %%
''' To investigate whether certain merchants are consistently associated with higher fraud,
we calculated the fraud rate, number of fraud cases, and total transactions for each merchant.
After filtering to merchants with at least 100 transactions (693 merchants), 
the top group showed fraud rates around 2–2.6%, which is several times higher than the overall fraud rate in the dataset (less than 1%).
This indicates that fraud is not evenly distributed across the ecosystem;
instead, it is concentrated among a relatively small set of high-risk merchants.
Based on this analysis, we created a new feature called merchant_risk_score,
which assigns each transaction the historical fraud rate of its merchant.
This feature captures merchant level risk and can be fed into our classification models to improve the detection of suspicious transactions.
'''
# %%
#Time-Based Fraud Patterns - SMART Question 2 

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
print(df['trans_date_trans_time'].dtype)

# %%
#Extracting time-based features

df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek  # 0=Monday, 6=Sunday
df['month'] = df['trans_date_trans_time'].dt.month

print(df[['hour', 'day_of_week', 'month']].head())

# %%
#Fraud vs Legit Transactions by Hour of the Day

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='hour', hue='is_fraud')
plt.title('Fraud vs Legit Transactions by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Transactions')
plt.tight_layout()
plt.show()

# %%
#Fraud vs Legit Transactions by Day of the Week

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='day_of_week', hue='is_fraud')
plt.title('Fraud vs Legit Transactions by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Transactions')
plt.tight_layout()
plt.show()

# %%
#Fraud vs Legit Transactions by Month

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='month', hue='is_fraud')
plt.title('Fraud vs Legit Transactions by Month')
plt.xlabel('Month')
plt.ylabel('Number of Transactions')
plt.tight_layout()
plt.show()

# %%
'''
To investigate whether fraudulent transactions occur more frequently at specific times of the day,
we extract time-based features from the 'trans_date_trans_time' column, including:
- Hour of the day: This helps in identifying if fraud is more likely at certain hours.
- Day of the week: To check if fraud occurs more frequently on weekends or weekdays.
- Month: To examine if there are seasonal patterns that affect the likelihood of fraud.

After extracting these time features, we:
1. Plot the distribution of fraudulent and legitimate transactions for each time-based feature:
   - Fraud vs Legit Transactions by Hour of the Day
   - Fraud vs Legit Transactions by Day of the Week
   - Fraud vs Legit Transactions by Month
2. Calculate the fraud rate (average fraud rate) for each hour, day, and month to identify if fraud is more likely during specific periods.
'''

# %%
"""
We visualize the distribution of fraudulent vs legitimate transactions based on **hour of the day**. This helps us see if fraud is more likely during specific hours. 
We calculate the **fraud rate** for each hour of the day, each day of the week, and each month to investigate time-based fraud patterns.

"""

#Calculate fraud rates by hour of the day
hourly_fraud_rate = df.groupby('hour')['is_fraud'].mean()

#Calculate fraud rates by day of the week
daily_fraud_rate = df.groupby('day_of_week')['is_fraud'].mean()

#Calculate fraud rates by month
monthly_fraud_rate = df.groupby('month')['is_fraud'].mean()

#Display the calculated fraud rates
print("\nHourly Fraud Rate:\n", hourly_fraud_rate)
print("\nDaily Fraud Rate:\n", daily_fraud_rate)
print("\nMonthly Fraud Rate:\n", monthly_fraud_rate)


# %%
"""
We conduct a **Chi-Square test** to examine if the **hour of the day** has a significant relationship with fraud occurrence.
"""

from scipy.stats import chi2_contingency

# Create contingency table for Day of the Week and Fraud
contingency_table = pd.crosstab(df['day_of_week'], df['is_fraud'])

# Perform Chi-Square Test
chi2, p_val, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square Test: p-value = {p_val}")

#%%
#contingency table for Hour of the Day and Fraud
contingency_table_hour = pd.crosstab(df['hour'], df['is_fraud'])

#Chi-Square Test
chi2_hour, p_val_hour, dof_hour, expected_hour = chi2_contingency(contingency_table_hour)
print(f"Chi-Square Test for Hourly Fraud Patterns: p-value = {p_val_hour}")

# %%
#contingency table for Month and Fraud
contingency_table_month = pd.crosstab(df['month'], df['is_fraud'])

#Chi-Square Test
chi2_month, p_val_month, dof_month, expected_month = chi2_contingency(contingency_table_month)
print(f"Chi-Square Test for Monthly Fraud Patterns: p-value = {p_val_month}")

# %%
'''
This section performs Chi-Square tests to assess the relationship between fraud and time-based features:
1. Hour of the Day: A Chi-Square test is conducted to check if the likelihood of fraud is significantly associated with specific hours of the day.
2. Day of the Week: A Chi-Square test is used to investigate whether fraud is more likely on specific days of the week (e.g., weekends or weekdays).
3. Month of the Year: A Chi-Square test is applied to examine if fraud is concentrated in certain months.

The resulting p-values indicate:
- Hour: The p-value of 0.0 suggests a strong association between fraud and the time of day.
- Day of the Week: The p-value of 2.29e-37 indicates a significant relationship between fraud and the day of the week.
- Month: The p-value of 6.25e-93 highlights a strong statistical relationship between fraud and the month of the year.

'''
# %%
"""
We visualize the **fraud rate by transaction category** using a **bar plot**. This helps identify categories that are more prone to fraud.
"""

# Transaction Amount & Category Analysis 
# Fraud Rate by Category
fraud_rate = df.groupby("category")["is_fraud"].mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(
    x=fraud_rate.index,
    y=fraud_rate.values,
    order=fraud_rate.index,
    color="salmon"
)
plt.xticks(rotation=45)
plt.ylabel("Fraud Rate")
plt.title("Fraud Rate by Category")
plt.tight_layout()
plt.show()

'''
The analysis shows that the categories Shopping, Misc_net, and Grocery_pos have the highest fraud rates, 
meaning they experience the most frequent fraudulent transactions relative to their total transactions. 
This suggests these categories may require closer monitoring or additional fraud prevention measures.
'''

#Spend Amount Distribution Across Fraud & Non-Fraud Transactions
plt.figure(figsize=(8,5)) 
sns.boxplot( 
    data=df,
    x="is_fraud",
    y="amt",
    palette= ["lightgreen", "salmon"]
 ) 
plt.title("Amount Distribution: Fraud vs Non-Fraud") 
plt.xticks([0,1], ["Non-Fraud", "Fraud"]) 
plt.show()

'''
The analysis shows that non-fraud transactions span a wide range of amounts with extreme high-value outliers,
while fraud transactions are concentrated within a narrower range. This suggests that fraud tends to occur within specific amounts,
whereas non-fraud transactions are more variable.
'''

#%%
'''
Do transaction amounts and categories significantly differ between fraud and non-fraud transactions?
I will perform the following statistical tests:
1) Independent Samples t-Test for transaction amounts
2) Chi-Square Test for transaction categories
Independent t-Test: Checks if transaction amounts differ significantly 
between fraud and non-fraud transactions.
'''
'''
1)Independent Samples t-Test for transaction amounts
Hypothesis:
    Null hypothesis (H₀): There is no difference in the mean transaction amount between fraud and non-fraud transactions.
    Alternative hypothesis (H₁): There is a difference in the mean transaction amount between fraud and non-fraud transactions.
'''

fraud_amt = df[df["is_fraud"] == 1]["amt"]
non_fraud_amt = df[df["is_fraud"] == 0]["amt"]

t_stat, p_value_amt = ttest_ind(fraud_amt, non_fraud_amt, equal_var=False)

print("\n Independent Samples t-Test: Amount vs Fraud")
print("t-statistic:", t_stat)
print("p-value:", p_value_amt)

'''
Since the p-value is less than 0.05, we reject the null hypothesis.
This indicates that the mean transaction amount for fraud transactions is significantly different from non-fraud transactions.

'''
'''
2) Chi-Square Test for transaction categories
Hypotheses:
    Null hypothesis (H₀): Fraud occurrence is independent of transaction category.
    Alternative hypothesis (H₁): Fraud occurrence is dependent on transaction category.
'''
contingency = pd.crosstab(df["category"], df["is_fraud"])
chi2, p_value_cat, dof, expected = chi2_contingency(contingency)

print("\n Chi-Square Test: Category vs Fraud")
print("Chi-Square:", chi2)
print("p-value:", p_value_cat)

'''
Since the p-value is less than 0.05, we reject the null hypothesis.
This indicates that fraud is not independent of transaction category, where certain categories are much more likely to experience fraudulent transactions.
This aligns with the earlier analysis showing higher fraud rates in categories like Shopping, Misc_net, and Grocery_pos.
'''

#%% Feature Selection 
model_df = df[[
    "amt",
    "category",
    "merchant_risk_score",
    "hour",
    "day_of_week",
    "month",
    "lat", "long",
    "merch_lat", "merch_long",
    "is_fraud"
]]


#%%
model_df = pd.get_dummies(model_df, columns=["category"], drop_first=True)


#%%Train/Test Split
from sklearn.model_selection import train_test_split

X = model_df.drop("is_fraud", axis=1)
y = model_df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


#%%
class_weights = {0:1, 1:20}  


#%%
# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]


#%%
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))

'''
The model correctly predicted 318,556 normal transactions
It mistakenly marked 3,736 normal transactions as fraud
It correctly caught 1,716 fraud transactions
It missed only 161 fraud transactions

The model is really good at catching fraud and only misses a small number of them.
It does mark some normal transactions as fraud, but that’s expected because fraud is rare and we trained the model to be more careful.



For fraud (1):
Precision = 0.31:
When the model says “fraud,” it is right about 31% of the time.

Recall = 0.91:
The model catches 91% of all actual fraud cases, which is very good.

F1 Score = 0.47:
This is a balance between catching fraud and being correct when predicting it.

Accuracy is 99%, but that's just because most transactions are normal.
The most important part is the high recall, which means the model is great at finding fraud.

The model catches most fraud cases, even if it occasionally raises a false alarm.
This is exactly what we want in fraud detection.Our ROC-AUC score is 0.993, which is very close to perfect.

The model is excellent at telling fraud apart from normal transactions.
'''

#%%
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(15).plot(kind='bar', figsize=(10,5))
plt.title("Top Feature Importances - Random Forest")
plt.show()
'''
The most important features the model used were:
Transaction amount
Hour of the transaction
Merchant risk score (fraud history of the merchant)
Transaction category
Location differences between customer & merchant

The model mostly looks at how much money was spent, the time of the transaction, and whether the merchant has a history of fraud.
'''

#%%
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Model selection
model_df = df[[
    "amt", 
    "merchant_risk_score", 
    "hour", 
    "day_of_week", 
    "month", 
    "category",
    "lat", "long", "merch_lat", "merch_long", 
    "is_fraud"
]]

#convert categorical columns to dummy variables -one-hot encoding for 'category'
model_df = pd.get_dummies(model_df, columns=["category"], drop_first=True)

#Features and target
X = model_df.drop("is_fraud", axis=1)
y = model_df["is_fraud"]

#Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#%%
#train-test split
#75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

#%%
#model
log_reg_model = LogisticRegression(class_weight='balanced', random_state=42)
log_reg_model.fit(X_train, y_train)

#%%
#predictions
y_pred = log_reg_model.predict(X_test)
y_prob = log_reg_model.predict_proba(X_test)[:, 1]  # Probability for ROC-AUC
#%%
#evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_prob))




# %%
"""
Results Interpretation for Logistic Regression Model:

1. **Confusion Matrix**:
   - The model correctly identified 283,742 legitimate transactions and 1,438 fraudulent ones.
   - However, 38,550 legitimate transactions were incorrectly flagged as fraud (false positives), which could lead to **unnecessary fraud alerts** or customer inconvenience.
   - The model missed 439 fraud cases (false negatives), meaning some fraudulent transactions were not detected.

2. **Classification Report**:
   - **Precision for Legit Transactions (0)**: The model is highly accurate (99%) when identifying legitimate transactions, meaning most transactions it flags as legit are truly legitimate.
   - **Recall for Fraudulent Transactions (1)**: The model correctly identifies 77% of fraudulent transactions, which is **good** but leaves room for improvement in detecting more fraud.
   - **Precision for Fraudulent Transactions (1)**: The model only flags fraud accurately 4% of the time, which means it incorrectly flags many legitimate transactions as fraud (high false positive rate).
   - **F1-Score**: For fraud detection, the F1-score is **low (0.07)**, indicating that the model needs improvement in finding fraud while reducing false alarms.

3. **ROC-AUC Score**:
   - A **score of 0.92** indicates that the model is **very effective** at distinguishing between fraud and legitimate transactions.

"""   