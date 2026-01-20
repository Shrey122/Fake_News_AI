# Fake_News_AI

Fake News Detection Project
This project focuses on detecting fake news using the LIAR dataset. It includes data preprocessing, exploratory data analysis (EDA), feature extraction, and the implementation and evaluation of several machine learning models.

Project Structure
fake_news_final.ipynb: The main Jupyter Notebook containing the complete workflow from data loading to model evaluation and visualization.
fake_news.ipynb: An earlier version or alternative approach to the fake news detection task.
combine_dataset.ipynb: A utility notebook to combine the different parts of the LIAR dataset (train, test, valid) into a single CSV file.
liar_dataset/: This directory contains the LIAR dataset files.
liars_dataset.csv: The combined dataset created by combine_dataset.ipynb.
README: The original README for the LIAR dataset.
test.tsv, train.tsv, valid.tsv: The original dataset files.
*.joblib: Saved machine learning models and the label encoder.
model_metrics.json: Stores the performance metrics of the trained models.
visualization_data.json: Data used for generating visualizations of model performance and analysis.
requirements.txt: A list of Python dependencies required to run the project.
.gitignore: Specifies intentionally untracked files that Git should ignore.
Objective
The primary objective of this project is to develop a robust system for classifying news statements into six categories of truthfulness based on the LIAR dataset. This involves:

Loading and preprocessing the text data.
Performing EDA to understand data characteristics.
Extracting features using TF-IDF.
Training and evaluating machine learning models: Logistic Regression, Random Forest, and Support Vector Machines (SVM).
Comparing model performance using metrics like accuracy, precision, recall, and F1-score.
Dataset
The project utilizes the LIAR dataset, which contains over 12,000 manually labeled short statements. The labels are:

Pants-on-Fire
False
Barely-True
Half-True
Mostly-True
True
For more details on the dataset, refer to liar_dataset/README.

How to Run
Clone the repository:

git clone https://github.com/OtoYuki/fake-news-detection-liar.git
cd fake-news-detection-liar
Set up a Python virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required dependencies:

pip install -r requirements.txt
Run the Jupyter Notebooks:

Start Jupyter Lab or Jupyter Notebook:

jupyter lab
# or
jupyter notebook
Open and run the cells in combine_dataset.ipynb first to generate liars_dataset.csv.

Then, open and run the cells in fake_news_final.ipynb to see the main analysis and model training.

Models and Evaluation
The following models were implemented and evaluated:

Logistic Regression
Random Forest
Support Vector Machine (SVM)
Performance metrics (accuracy, precision, recall, F1-score, training time) for each model are stored in model_metrics.json and visualized in the fake_news_final.ipynb notebook. The Random Forest model generally performed the best.
