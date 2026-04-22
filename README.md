# Comparative Analysis of Neural Network Architectures for Image Classification

## Problem Statement

The task is to classify iris flowers into three species:

- Setosa  
- Versicolor  
- Virginica  

based on four numerical features:

- Sepal Length  
- Sepal Width  
- Petal Length  
- Petal Width  

This is a classic supervised learning problem and serves as a good starting point to understand model behavior, feature relationships, and decision boundaries.

---

## Dataset

- **Dataset used:** Iris Dataset  
- **Total samples:** 150  
- **Classes:** 3  
- **Features:** 4 numerical attributes  

---

### Key Observations

- Classes are fairly well-separated (especially Setosa)  
- Some overlap exists between Versicolor and Virginica, making classification slightly challenging  

---

## Exploratory Data Analysis (EDA)

A pairplot was used to visualize feature relationships and class separability:

```python
sns.pairplot(df, hue='Species')
```

This helps in understanding:

- Feature distribution
- Class clustering
- Potential linear separability

## Data Preprocessing

- Dropped unnecessary column: Id
- Encoded labels using LabelEncoder
- Standardized features using StandardScaler
- Used stratified train-test split (80-20) to maintain class balance

## 🤖 Models Used

### 1. Perceptron (Baseline Model)

A simple linear classifier used as a baseline to understand how well a linear decision boundary performs.

**Configuration:**
- Max iterations: 1000  
- Random state: 42  

**Performance:**
- Accuracy: **86.67%**

This shows that while the dataset is somewhat linearly separable, there are limitations.

---

### 2. Artificial Neural Network (ANN)

A feedforward neural network built using TensorFlow/Keras to capture non-linear relationships.

**Architecture:**
- Input Layer: 4 features  
- Hidden Layer 1: 16 neurons (ReLU)  
- Hidden Layer 2: 8 neurons (ReLU)  
- Output Layer: 3 neurons (Softmax)  

```python
Dense(16, activation='relu')
Dense(8, activation='relu')
Dense(3, activation='softmax')
```

**Training Setup:**
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Epochs: 100  
- Batch Size: 8  
- Validation Split: 20%  

---

## 📈 Results

| Model       | Accuracy  |
|------------|----------|
| Perceptron | 86.67%   |
| ANN        | ~95–98%  |

The ANN significantly outperforms the Perceptron due to its ability to model non-linear relationships.

---

## 📉 Training Visualization

Training and validation accuracy were plotted to monitor performance:

```python
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
```

This helps in:

- Detecting overfitting  
- Understanding convergence behavior  

---

## 🧠 Key Learnings

- Linear models like Perceptron have limited capacity for complex boundaries  
- Neural networks can significantly improve performance even on small datasets  
- Feature scaling plays a critical role in model convergence  
- Stratified splitting ensures fair evaluation across classes  
- Visualization (EDA + training curves) is essential for debugging and understanding models  

---

## 📌 Future Improvements

- Add Dropout layers to study overfitting behavior  
- Try deeper architectures or different activation functions  
- Compare with other models (SVM, Random Forest, etc.)  
- Perform hyperparameter tuning  
- Visualize decision boundaries  

