#Import Necessary Libraries:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load the Dataset
df = pd.read_csv('Titanic-Dataset - Titanic-Dataset.csv')
df.head()

#Check for Missing Values:
df.isnull().sum()

#Handle Missing Values (Drop or Impute):
df['Age'].fillna(df['Age'].median(), inplace=True)

#Descriptive Statistics:
df.describe()

#Visualize Distributions:
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution of Passengers')
plt.show()

sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

#Analyze Correlations:
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Matrix')
plt.show()

#Survival Analysis Based on Class:
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Class')
plt.show()


