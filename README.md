# Credit Card Fraud Detection Using Machine Learning

### **Subtitle**: Analyzing Transaction Patterns and Developing Predictive Models for Real-Time Fraud Detection

## Project Overview
This project focuses on building a **machine learning-based fraud detection system** to identify fraudulent credit card transactions in real-time. Using a dataset sourced from **Kaggle**, which contains simulated transactions from **January 2019 to December 2020**, the project analyzes patterns in **transaction amounts**, **merchant information**, **transaction timing**, and **customer demographics** to develop predictive models. The objective is to improve fraud detection accuracy and help financial institutions minimize financial losses caused by fraudulent transactions.

## SMART Research Questions
1. **Merchant-Level Risk**: Are certain merchants consistently associated with higher fraud rates, and can merchant-level risk scores improve model prediction?
2. **Time-Based Fraud Patterns**: How does the time of transaction affect the likelihood of fraud, and can we identify specific time patterns?
3. **Spending Behavior and Fraud**: Is there a significant relationship between spend amount, category, and transaction fraud prediction?

## Dataset
The dataset used for this analysis is the **Credit Card Fraud Detection Dataset** sourced from **Kaggle**.

- **Duration**: **January 2019 – December 2020**
- **Number of Transactions**: **~555,718**
- **Number of Customers**: **~1,000**
- **Number of Merchants**: **~800**
- **Features**: Includes transaction amounts, merchant details, customer demographics, and fraud labels.

**Dataset Link**: [Credit Card Fraud Detection Dataset - Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

## Business Insights
The **Business Insights** document in **PDF format** is included in this repository. It contains actionable business recommendations based on the results of the analysis. The insights focus on fraud detection strategies related to **merchant-level risk**, **transaction timing**, and **spending behavior** to help optimize fraud detection systems.

You can access the insights document here: [**Business Insights: Credit Card Fraud Detection Analysis**](./Business%20Insights%3A%20Credit%20Card%20Fraud%20Detection%20Analysis.pdf).


## Requirements
To run this notebook, you'll need Python and the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
