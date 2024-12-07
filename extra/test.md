
# Financial Product Recommendation

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

## Next Steps

While the models provide reasonable accuracy, there are several ways to improve the project:
1. **Model Improvement**: Implement more advanced models like XGBOost,Random Forest or Gradient Boosting.
2. **Hyperparameter Tuning**: Fine-tune the models' hyperparameters using Grid Search or Random Search to improve performance.
3. **Feature Selection**: Perform feature selection or dimensionality reduction to reduce model complexity and prevent overfitting.
4. **Business Application**: Integrate the predictive model into a larger system that recommends financial products based on an individual’s predicted income bracket.

