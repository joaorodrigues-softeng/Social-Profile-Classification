# Social Profile Classification and Analysis Using Python

## Overview
This project focuses on analyzing and classifying social profiles based on a dataset collected from a study on decision-making criteria during the COVID-19 pandemic. The dataset includes various attributes such as gender, age, marital status, education level, professional situation, political orientation, and decision-making classification. The goal is to create a Python application that allows users to analyze and classify these profiles via terminal commands.

## Dataset
The dataset `saúde.csv` contains the following columns:
- **Gender**: Male or Female
- **Age**: Age of the respondent
- **Marital Status**: Marital status of the respondent
- **Education Level**: Highest level of education achieved
- **Professional Situation**: Current employment status
- **Political Orientation**: Political leaning (Left, Center, Right, or No Option)
- **Decision-Making Classification**: Spontaneous, Intuitive, Dependent, Avoidant, Rational

## Project Structure
The project is structured into four main parts:

1. **Data Frame Creation (made with saude.ipynb)**:
   - Created a new DataFrame with specific columns and save it as a new CSV file.
   - Converted age into generational categories.
   - Normalized political orientation to a scale of 0-3.
   - Filled missing values for marital status, education level, and professional situation with 0.
   - Converted decision-making classes into binary values based on the highest score.

2. **Exploratory Analysis**:
   - Calculate the percentage of men and women in each decision-making class.
   - Perform similar analyses for marital status, education level, professional situation, and political orientation.
   - Conduct additional interesting analyses based on the dataset.

3. **Data Visualization**:
   - Create graphical analyses that could be included in a potential dashboard.
   - Construct a heatmap and draw conclusions from it.

4. **Classification**:
   - Classify the "Rational" class using a simple classification method and analyze the results.
   - Optional challenge: Research and implement a multi-class classification algorithm to identify multiple decision-making traits (e.g., Spontaneous and Dependent).

## How to Use
The project is designed to be user-friendly with a terminal-based menu. The user can select from the following options:
1. **Exploratory Analysis**:
   - Perform various exploratory analyses on the dataset.
2. **Graphical Analysis**:
   - Visualize the data using graphs and heatmaps.
3. **Predictive Analysis**:
   - Classify the "Rational" class and optionally implement multi-class classification.

## Requirements
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (for classification tasks)

## How to Run the Project
1. Use `saudefinal.csv` with the python code provided.
2. Import main.py in PyCharm.

## Technologies Used
- Pycharm
- Jupyter Lab

## Author
João Rodrigues (joaorodrigues.softeng@gmail.com)

## License
This project is open-source and available under the MIT License.
