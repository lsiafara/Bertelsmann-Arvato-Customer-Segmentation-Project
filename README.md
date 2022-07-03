# Bertelsmann-Arvato-Customer-Segmentation-Project
### Table of Contents
1. [Summary](#summary)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Overview](#overview)
5. [Results](#results)

## Summary <a name="summary"></a>
This project analyzes demographic data from Bertelsmann Arvato to build a model that predicts which individuals are most likely to respond to the marketing campaign and become new clients of a mail-order company. Specifically, it creates a machine learning pipeline to predict whether the response of each person will be positive or negative.

## Installation <a name="installation"></a>

To install the required packages run the code below
 'pip install -r requirements.txt'

### File Descriptions <a name="files"></a>

1. Jupyter Notebook
</br>There is 1 jupyter notebook file 'Arvato Project Workbook.ipynb' with the python script used for the analysis and markup text explaining each step.

2. Python file
</br>There is 1 python file 'Arvato_Project.py' with the code used to read the data, load the data, cluster the data and predict the response for the test set.

3. Kaggle CSV file
</br>There is 1 csv file 'kaggle_results.csv' with 2 columns: the ID number for each individual in the test set and the predicted response.

## Overview <a name="overview"></a>
The analysis consists of 3 parts:
  1. Data preprocessing
  2. Unsupervised learning for customer segmenation
  3. Supervised learning for prediction of the response rate

### Results <a name="results"></a>
The analysis steps as well as the results have been documented in the Medium blog [Report for the Bertelsmann-Arvato Project](https://medium.com/@lydia.siafara/report-for-the-bertelsmann-arvato-project-f89c3e783773).
