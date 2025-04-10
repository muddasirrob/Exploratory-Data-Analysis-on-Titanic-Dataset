# **Exploratory Data Analysis on Titanic Dataset**

## **Project Overview**

This project performs **Exploratory Data Analysis (EDA)** on the **Titanic dataset** to uncover insights into the factors influencing passenger survival. The dataset contains information such as passenger age, sex, class, and whether they survived the Titanic disaster. The goal is to explore key patterns and trends in the data, visualize these patterns, and derive conclusions about the various factors that impacted survival rates.

### **Key Objectives:**
- **Data Cleaning**: Handle missing values, convert categorical variables into numerical ones, and ensure the dataset is ready for analysis.
- **Exploratory Analysis**: Calculate and display summary statistics, distribution of features, and correlations between various variables.
- **Visualization**: Generate visualizations to help identify important trends and insights, such as survival rates across different categories.
- **Insight Extraction**: Identify which features (e.g., sex, class, age) significantly contributed to the survival of passengers.

## **Technologies Used**
- **Python**
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization and plotting.
- **Seaborn**: For advanced statistical plotting and heatmaps.
- **Jupyter Notebook**: For creating and executing the analysis interactively.

## **Dataset**
The dataset used for this project is the **Titanic dataset**, which is publicly available and often used for educational purposes in data science and machine learning. The dataset includes columns such as:
- `Survived`: Whether the passenger survived (1 = Yes, 0 = No).
- `Pclass`: The passenger class (1, 2, or 3).
- `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.

## **Steps Involved**
1. **Data Loading**: Load the Titanic dataset and display basic info.
2. **Data Cleaning**: Handle missing values and encode categorical variables.
3. **Descriptive Statistics**: Compute summary statistics for numerical features.
4. **Data Visualization**: Create plots to examine survival rates by gender, class, age, and other factors.
5. **Survival Analysis**: Analyze how different features impact survival.

## **How to Use**
1. Clone the repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install pandas matplotlib seaborn
