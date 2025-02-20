# Predict Health Costs with Regression

## 📌 Overview
This project predicts medical insurance costs using a regression model built with TensorFlow and Keras. The dataset includes demographic and lifestyle factors such as age, sex, BMI, smoking habits, and region. The goal is to develop a model that generalizes well and achieves a Mean Absolute Error (MAE) below 3500.

## 📊 Dataset
- Source: [FreeCodeCamp Insurance Dataset](https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv)
- Features:
  - `age`: Age of the policyholder
  - `sex`: Gender (male/female)
  - `bmi`: Body Mass Index (BMI)
  - `children`: Number of dependents
  - `smoker`: Smoking status (yes/no)
  - `region`: Geographic region (northeast, northwest, southeast, southwest)
  - `expenses`: Medical insurance cost (Target Variable)

## ⚙️ Technologies & Skills Used
- **Machine Learning**: Regression Modeling
- **Deep Learning**: Neural Networks with TensorFlow & Keras
- **Data Preprocessing**: Handling categorical variables, one-hot encoding
- **Python Programming**: NumPy, Pandas, Matplotlib
- **Model Evaluation**: MAE, MSE, Loss Functions
- **Optimization**: Adam Optimizer

## 🚀 Project Workflow
1. **Data Preprocessing**:
   - Convert categorical data to numerical values
   - Perform one-hot encoding for region
   - Split the dataset into training (80%) and testing (20%) sets
2. **Model Architecture**:
   - 3-layer Neural Network with ReLU activation
   - Optimized with Adam and Mean Squared Error (MSE)
3. **Training & Evaluation**:
   - Train for 100 epochs with validation split
   - Evaluate performance on test data
   - Aim for MAE < 3500

## 📈 Results
- The model achieves an MAE of approximately `XXXX` (to be updated based on training).
- Scatter plot comparison of true vs. predicted values shows the model's accuracy.

## 🔧 How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow pandas numpy matplotlib
   ```
2. Run the Python script or Jupyter Notebook:
   ```python
   python predict_health_costs.py
   ```
3. Evaluate model performance and visualize results.

## 📌 Future Improvements
- Feature engineering to enhance predictive performance
- Experimenting with different architectures (e.g., deeper networks, regularization)
- Hyperparameter tuning for optimization

## 🏆 Acknowledgments
- FreeCodeCamp for providing the dataset
- TensorFlow/Keras for deep learning framework

---
### 🌟 **Contributions**
Feel free to fork the repository and contribute improvements. If you have suggestions, open an issue or submit a pull request!
