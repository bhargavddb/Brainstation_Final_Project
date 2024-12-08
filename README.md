
# Financial Product Recommendation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
   - [Part 1: LOAD | CLEAN | TRANSFORM](#1-part-1-load--clean--transform)
   - [Part 2: EDA (Exploratory Data Analysis)](#2-part-2-eda-exploratory-data-analysis)
   - [Part 3: PREPROCESSING | ML TEST](#3-part-3-preprocessing--ml-test)
3. [Project Workflow](#project-workflow)
4. [Installation and Dependencies](#installation-and-dependencies)
5. [How to Use](#how-to-use)
6. [Dataset](#dataset)
7. [Data Source](#data-source)
8. [Results and Discussion](#results-and-discussion)
   - [Key Findings](#key-findings)
9. [Final Product app (   Secure Financial Recommendation System - Patron )](#final_app)


## Project Overview

This project aims to build a machine learning model that predicts an individual's income bracket based on demographic and employment-related features such as age, work class, education, and hours worked per week. The income brackets are divided into two categories: `<= 50K` and `> 50K`. The ultimate goal of this prediction is to recommend suitable financial products to individuals, such as stocks, credit cards, or loan offers based on their income group.

The project encompasses multiple stages, starting from data cleaning and exploratory analysis to machine learning model development and evaluation. It focuses on understanding the socio-economic factors influencing income distribution and leverages this understanding to develop personalized financial product recommendations.

## Repository Structure

This repository is organized into three key Jupyter notebooks, each focusing on different aspects of the project pipeline:

### 1. **Part 1: LOAD | CLEAN | TRANSFORM**
   - **Purpose**: This notebook is responsible for loading the dataset, performing data cleaning, and transforming the data to ensure it's ready for analysis and machine learning.
   - **Key Steps**:
     - **Loading Data**: The dataset is loaded from a CSV file and basic information about the data is explored.
     - **Data Cleaning**: This includes handling missing values, outliers, and correcting inconsistencies in categorical data.
     - **Numerical Outliers**: We identify and handle outliers in the numerical columns to prevent them from skewing the analysis.
     - **Categorical Data Cleaning**: Text data is cleaned and transformed for easier processing.
     - **Saving the Cleaned Data**: The processed dataset is saved as a CSV file for use in further analysis and model building.

### 2. **Part 2: EDA (Exploratory Data Analysis)**
   - **Purpose**: This notebook performs detailed exploratory data analysis (EDA) to uncover insights about the dataset and identify key relationships between features.
   - **Key Steps**:
     - **Age Distribution**: Analyze the age distribution across different income groups to identify any patterns.
     - **Income Distribution**: A detailed analysis of income distribution within the dataset, visualizing how different factors influence income.
     - **Education and Workclass**: Explore the relationship between education level, work class, and income, and see how education impacts earning potential.
     - **Capital Gain and Loss**: Investigate how capital gain and loss vary between different income groups.
     - **Hours Worked vs Income**: Analyze the correlation between the number of weekly work hours and income categories.
     - **Martial Status vs Income**: Investigate how Married Civil Spouse status has higher chance of Income.
     - **Occupation vs Income**: Investigate how the Exceutive Job titles has higher chance of Income.
     - **Statistical Analysis**: Various statistical tests (e.g., Chi-square test, T-tests) are performed to check for significant relationships between features and the income bracket.

### 3. **Part 3: PREPROCESSING | ML TEST**
   - **Purpose**: This notebook focuses on preparing the data for machine learning models and testing different algorithms to predict the income category of individuals.
   - **Key Steps**:
     - **Preprocessing**: Based on insights from EDA, features are engineered and prepared for input into machine learning models. This includes scaling numerical features, encoding categorical variables, and creating train/test splits.
     - **Machine Learning Algorithms**:
       - **K-Nearest Neighbors (KNN)**: A simple, intuitive algorithm that classifies individuals based on the 'k' nearest neighbors in the dataset.
       - **Logistic Regression**: A probabilistic model that estimates the likelihood of a person falling into a particular income bracket based on their features.
       - **Decision Trees**: A model that splits the data into branches based on feature values, offering high interpretability.
     - **Model Evaluation**: The models are evaluated using performance metrics such as accuracy, precision, recall, and F1-score. Confusion matrices and classification reports provide deeper insights into model performance.
     - **Key Findings**: Analysis of the model results, discussing the accuracy and reliability of the predictions, and identifying areas for improvement.

## Project Workflow

1. **Data Loading and Initial Cleaning**: The dataset is loaded, basic cleaning is performed, and the data is saved for further exploration.
2. **Exploratory Data Analysis (EDA)**: The cleaned data is explored to gain insights into feature distributions and relationships.
3. **Feature Engineering and Preprocessing**: Key features are selected and prepared for input into machine learning.
4. **Model Training and Testing**: Various machine learning algorithms are applied to predict the income category.
5. **Evaluation and Next Steps**: The models are evaluated, and further improvements are suggested.

## Installation and Dependencies

To replicate this project, you’ll need to have Python 3.x installed on your machine along with several libraries. The primary libraries used in this project are:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `statsmodels`
- `scipy`
- `from sklearn.model_selection import train_test_split`
- `from sklearn.preprocessing import StandardScaler`
- `from sklearn.tree import DecisionTreeClassifier`
- `from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,ConfusionMatrixDisplay`
- `from sklearn.neighbors import KNeighborsClassifier`
- `from sklearn.linear_model import LogisticRegression`
- `from sklearn.ensemble import RandomForestClassifier`

You can install these dependencies using `pip`:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn statsmodels scipy sklearn
```

Additionally, you'll need to install Jupyter Notebook or JupyterLab to run the notebooks. Install it using:

```bash
pip install notebook
```

## How to Use

1. **Clone the repository** to your local machine:

   ```bash
   git clone https://github.com/bhargavddb/FINAL_PROJECT.git
  
   ```

2. **Open the notebooks** using Jupyter:

   ```bash
   jupyter notebook
   ```

3. **Run the Notebooks**:
   - Start with **Part 1: LOAD | CLEAN | TRANSFORM** to preprocess the dataset.
   - Proceed with **Part 2: EDA** to explore the dataset and discover patterns.
   - Finally, execute **Part 3: PREPROCESSING | ML TEST** to preprocess the data, train machine learning models, and evaluate their performance.

## Dataset

The dataset used in this project is derived from the U.S. Census Bureau and is commonly referred to as the **Census Income Dataset**. It contains the following features:

<table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th style="border: 1px solid #ddd; padding: 8px; font-weight: bold; text-align: left;">Feature</th>
      <th style="border: 1px solid #ddd; padding: 8px; font-weight: bold; text-align: left;">Type</th>
      <th style="border: 1px solid #ddd; padding: 8px; font-weight: bold; text-align: left;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">age</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Numerical</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Age of the individual</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="border: 1px solid #ddd; padding: 8px;">workclass</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Categorical</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Type of employment (e.g., Private, Self-emp-not-inc)</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">education_level</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Categorical</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Highest level of education completed (e.g., Bachelors, HS-grad)</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="border: 1px solid #ddd; padding: 8px;">education-num</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Numerical</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Number of years of education completed</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">marital-status</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Categorical</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Marital status (e.g., Never-married, Divorced)</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="border: 1px solid #ddd; padding: 8px;">occupation</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Categorical</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Type of occupation (e.g., Exec-managerial, Handlers-cleaners)</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">relationship</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Categorical</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Family relationship (e.g., Husband, Wife)</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="border: 1px solid #ddd; padding: 8px;">race</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Categorical</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Race of the individual (e.g., White, Black)</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">sex</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Categorical</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Gender of the individual (Male, Female)</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="border: 1px solid #ddd; padding: 8px;">capital-gain</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Numerical</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Capital gains from investments</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">capital-loss</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Numerical</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Capital losses from investments</td>
    </tr>
    <tr style="background-color: #f9f9f9;">
      <td style="border: 1px solid #ddd; padding: 8px;">hours-per-week</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Numerical</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Number of hours worked per week</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">native-country</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Categorical</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Country of origin (e.g., United-States, Cuba)</td>
    </tr>
    <tr style="background-color: red;">
      <td style="border: 1px solid #ddd; padding: 8px;">income</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Categorical</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Income class (<=50K, >50K)</td>
    </tr>
  </tbody>
</table>

## Data Source
The dataset can be obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income) or similar sources.

## Results and Discussion

The project implemented several machine learning models, including:
- **K-Nearest Neighbors (KNN)**: Achieved decent accuracy but was computationally expensive for large datasets  ( 81.1 %  ).
- **Logistic Regression**: Provided interpretable results, highlighting which features most strongly influence income  ( 82.7 %  ).
- **Decision Tree Classifier**: Captured complex interactions between features and has simialar Accuracy of Logistic Regression  ( 83.3 %  ).

### Key Findings
- **Education and Income**: Higher education levels were strongly correlated with higher income.
- **Work Hours**: As expected, individuals working more hours per week had a higher likelihood of being in the `> 50K` category.
- **Capital Gains**: Individuals with substantial capital gains were more likely to be in the `> 50K` income group.

## Final Product app (   Secure Financial Recommendation System - Patron )

## Overview
The **Secure Financial Recommendation System - Patron** is a Streamlit-based application designed to recommend financial products to users based on their demographic data and predicted income category. This tool uses machine learning models and clustering techniques to provide tailored recommendations.
 link :- https://patron-bd.streamlit.app/
 username : guest
 password : guest

## Features
1. **Login System**: Secure user authentication with predefined credentials.
2. **Machine Learning Integration**: Logistic Regression and KMeans Clustering for predictions and insights.
3. **Financial Recommendations**: Tailored product suggestions, including credit cards, loans, savings plans, and investments.
4. **Interactive UI**: User-friendly input via the Streamlit sidebar.
5. **Dynamic Background and Styling**: Customizable landing page and main app view.

## Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- KMeans Clustering
- Logistic Regression
- Base64 Encoding (for images)

## Installation

### Prerequisites
- Python 3.8 or higher

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/bhargavddb/Brainstation_Final_Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd financial-recommendation-system
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the application:
   ```bash
   streamlit run Main_App.py
   ```
2. Open the web app in your browser at `http://localhost:8501`.
3. Login using the following credentials:
   - Username: `admin`
   - Password: `password123`

4. Enter your details in the sidebar to get financial product recommendations.

## Key Functionalities
### Data Loading
- Reads a preprocessed dataset (`census_modified.csv`) and prepares it for predictions.

### Model Training
- Logistic Regression for income prediction.
- KMeans Clustering to segment users into distinct clusters.

### Financial Product Mapping
Recommends financial products based on:
- Predicted income category (`<=50K` or `>50K`).
- Assigned cluster.

### Cluster Descriptions
Each cluster is associated with specific demographics and financial needs:
- **Cluster 0**: Entry-level professionals, basic savings needs.
- **Cluster 1**: Mid-level professionals, stable income.
- **Cluster 2**: Individuals needing credit repair.
- **Cluster 3**: High-income professionals focusing on investments.
- **Cluster 4**: Wealthy individuals investing in cryptocurrency.
- **Cluster 5**: Low-income earners preferring savings plans.

## Future Improvements
1. **Enhanced Security**: Use a database for user authentication.
2. **Additional Models**: Incorporate advanced models like XGBoost for better predictions.
3. **Dynamic Dataset**: Allow users to upload custom datasets.
4. **Real-Time Predictions**: Enable live updates for input changes.

## Screenshots
1. **Landing Page**: Custom background image with a login interface.
2. **Main App**: Sidebar for input and interactive result display.

## License
This project is licensed under the MIT License.







