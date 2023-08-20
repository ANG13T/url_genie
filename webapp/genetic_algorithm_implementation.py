'''
Optimized Malicious Domain Detection Model Generation Python Script
Created by Angelina Tsuboi (angelinatsuboi.com)

Objective:
This script was created with the goal of making a Multilayer Perceptron Neural Network optimized using genetic algorithms

The code sequence is as follows:
1. Integrate CSV Dataset and Remove Unecessary Columns
2. Use SMOTE to Balance out Class Distribution in Dataset
3. Split Dataset into Training and Testing Sets using 80:20 Ratio
4. Initialize Multilayer Perception
5. Utilize Adam Optimization and Binary Cross Entropy Loss Function
6. Initialize Model Callback to Wait Until 0.1 Validation Loss 
7. Train Model with 10 Epochs and Batch Size of 256
8. Verify Model Results using 10 Examples
9. Run Each Model Iteration through a Genetic Algorithm
10. Evaluate Fitness of Each Model by Referencing Accuracy
11. Determine Best Model within Population
12. Save the Best Model into a .h5 File Output
'''

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import keras
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import pandas as pd

# Configuring Dataset and Values

# Reading in dataset from CSV file. This dataset is an updated version of the original Kaggle dataset including 
urldata = pd.read_csv("./Url_Processed.csv")

# Clean up dataset and remove unecessary columns
urldata.drop("Unnamed: 0",axis=1,inplace=True)
urldata.drop(["url","label"],axis=1,inplace=True)

# Configure dependent variables (values used to inform prediction)
x = urldata[['hostname_length',
       'path_length', 'fd_length', 'count-', 'count@', 'count?',
       'count%', 'count.', 'count=', 'count-http','count-https', 'count-www', 'count-digits',
       'count-letters', 'count_dir', 'use_of_ip']]

# Configure independent variable (value to verify prediction)
y = urldata['result']

# Using SMOTE to resample dataset. The SMOTE (Synthetic Minority Over-sampling Technique) method is used to oversample the dataset
# SMOTE is used to balance the class distribution whenever it detects for an imbalance (one sample has signifcantly more samples than the other decreasing model performance) 
'''
Easy to understand example of SMOTE. Consider the following dataset with two features (X1 and X2) and a binary class label (y),
|  X1  |  X2  |  y  |
|------|------|-----|
|  1.5 |  2.0 |  0  |
|  2.0 |  3.0 |  0  |
|  3.0 |  5.0 |  1  |
|  3.5 |  4.5 |  0  |
|  4.0 |  3.5 |  0  |
|  4.5 |  4.0 |  0  |
|  5.0 |  2.5 |  1  |

There is an imbalance in the y column of the dataset as the class 1 is underrepresented.
If SMOTE is applied, the following output will be the result:
|  X1  |  X2  |  y  |
|------|------|-----|
|  1.5 |  2.0 |  0  |
|  2.0 |  3.0 |  0  |
|  3.0 |  5.0 |  1  |
|  3.5 |  4.5 |  0  |
|  4.0 |  3.5 |  0  |
|  4.5 |  4.0 |  0  |
|  5.0 |  2.5 |  1  |
| 2.75 | 4.25 |  1  |  # Synthetic sample (SMOTE)
| 4.25 | 2.75 |  1  |  # Synthetic sample (SMOTE)

As you can see, it generated two synthetic samples with the class 1 to balance out the class distribution in the dataset
'''
x_sample, y_sample = SMOTE().fit_resample(x, y.values.ravel())

x_sample = pd.DataFrame(x_sample)
y_sample = pd.DataFrame(y_sample)

# Seperate data into training and testing sets using the 80:20 ratio
x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size = 0.2)

# Define the hyperparameter search space
hyperparameter_space = {
    'num_hidden_layers': [1, 2, 3],
    'hidden_layer_units': [8, 16, 32, 64],
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [32, 64, 128, 256]
}

# Define the fitness function to evaluate the model
def evaluate_model(params):
    num_hidden_layers = params['num_hidden_layers']
    hidden_layer_units = params['hidden_layer_units']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    
    x_sample, y_sample = SMOTE().fit_resample(x, y.values.ravel())
    
    x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size=0.2)
    
    model = Sequential()
    model.add(Dense(hidden_layer_units, activation='relu', input_shape=(16,)))
    for _ in range(num_hidden_layers - 1):
        model.add(Dense(hidden_layer_units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    opt = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

    history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, verbose=0)
    
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    
    return f1

# Genetic algorithm parameters
population_size = 10
num_generations = 10
elite_size = 2

# Initialize population randomly
population = []
for _ in range(population_size):
    individual = {
        'num_hidden_layers': np.random.choice(hyperparameter_space['num_hidden_layers']),
        'hidden_layer_units': np.random.choice(hyperparameter_space['hidden_layer_units']),
        'learning_rate': np.random.choice(hyperparameter_space['learning_rate']),
        'batch_size': np.random.choice(hyperparameter_space['batch_size'])
    }
    population.append(individual)

# Genetic algorithm loop
for generation in range(num_generations):
    print(f"Generation {generation+1}/{num_generations}")
    
    # Evaluate individuals
    fitness_scores = []
    for individual in population:
        fitness = evaluate_model(individual)
        fitness_scores.append(fitness)
    
    # Select elite individuals
    elite_indices = np.argsort(fitness_scores)[-elite_size:]
    elite_population = [population[i] for i in elite_indices]
    
    # Create new generation
    new_population = elite_population.copy()
    while len(new_population) < population_size:
        parent1 = np.random.choice(elite_population)
        parent2 = np.random.choice(elite_population)
        child = {}
        for param in hyperparameter_space:
            if np.random.rand() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        new_population.append(child)
    
    population = new_population

# Find the best individual
best_fitness = max(fitness_scores)
best_individual = elite_population[np.argmax(fitness_scores)]
print("Best Fitness:", best_fitness)
print("Best Individual:", best_individual)


# Train and save the best model
best_model = Sequential()
best_model.add(Dense(best_individual['hidden_layer_units'], activation='relu', input_shape=(16,)))
for _ in range(best_individual['num_hidden_layers'] - 1):
    best_model.add(Dense(best_individual['hidden_layer_units'], activation='relu'))
best_model.add(Dense(1, activation='sigmoid'))

opt = keras.optimizers.Adam(lr=best_individual['learning_rate'])
best_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])

history = best_model.fit(x_train, y_train, epochs=10, batch_size=best_individual['batch_size'], verbose=0)

# Save the best model
best_model.save("Malicious_URL_Prediction.h5")

print("Best model saved as Malicious_URL_Prediction.h5")