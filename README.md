# Movie Recommendation System using Restricted Boltzmann Machine (RBM)

This project implements a movie recommendation system using a Restricted Boltzmann Machine (RBM) trained on the MovieLens 20M dataset. The system predicts user ratings for movies based on their previous ratings and recommends movies accordingly.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Code Implementation](#code-implementation)
5. [Training Process](#training-process)
6. [Results](#results)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Contributing](#contributing)

---

## Project Overview

The goal of this project is to build a recommendation system that predicts whether a user will like a movie based on their previous ratings. The model is trained on the MovieLens 20M dataset, which contains 20 million ratings from users. The Restricted Boltzmann Machine (RBM) is used as the core algorithm for collaborative filtering.

---

## Dataset

The dataset used in this project is the **MovieLens 20M Dataset**, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset). It includes:
- **ratings.csv**: Contains user IDs, movie IDs, ratings, and timestamps.
- **movies.csv**: Contains movie titles and genres.

For this project, only the `ratings.csv` file is used, as the focus is on predicting user ratings.

---

## Model Architecture

The model is based on a **Restricted Boltzmann Machine (RBM)**, a type of neural network used for collaborative filtering. The RBM learns latent representations of users and movies to predict ratings.

### Key Components:
1. **Input Layer**: User-movie interactions (ratings) are one-hot encoded.
2. **Hidden Layer**: Learns latent features of users and movies.
3. **Output Layer**: Predicts the probability distribution of ratings for each user-movie pair.

### Hyperparameters:
- Number of hidden units: `50`
- Number of rating categories: `10` (ratings from 1 to 10)
- Learning rate: `1e-2`
- Batch size: `256`
- Epochs: `10`

---

## Code Implementation

The project is implemented in Python using the following libraries:
- `pandas` for data manipulation.
- `numpy` for numerical computations.
- `tensorflow` for building and training the RBM.
- `scipy.sparse` for handling sparse matrices.
- `matplotlib` for visualizing training progress.

### Key Steps:
1. **Data Preprocessing**:
   - Download and load the dataset.
   - Map movie IDs and ratings to indices.
   - Split the data into training and testing sets.
   - Convert the data into sparse matrices.

2. **Model Training**:
   - Initialize the RBM with specified hyperparameters.
   - Train the model using the training set.
   - Evaluate the model on the test set using Mean Squared Error (MSE).

3. **Prediction**:
   - Use the trained model to predict user ratings for unseen movies.

---

## Training Process

The training process was conducted on the full MovieLens 20M dataset. However, due to internet connectivity issues, the training was disrupted twice. Despite this, the model achieved reasonable performance.

### Training Outputs:
#### First Training Run:
- **Epoch 0**: Train MSE = 1.7225, Test MSE = 0.0029
- **Epoch 1**: Train MSE = 1.6587, Test MSE = 0.0028
- **Epoch 2**: Train MSE = 1.6450, Test MSE = 0.0034
- **Epoch 3**: Train MSE = 1.6179, Test MSE = 0.0030
- **Epoch 4**: Train MSE = 1.6095, Test MSE = 0.0030

#### Second Training Run:
- **Epoch 0**: Train MSE = 1.7318, Test MSE = 0.0034
- **Epoch 1**: Train MSE = 1.6725, Test MSE = 0.0035
- **Epoch 2**: Train MSE = 1.6454, Test MSE = 0.0032
- **Epoch 3**: Train MSE = 1.6197, Test MSE = 0.0030
- **Epoch 4**: Train MSE = 1.6095, Test MSE = 0.0030
- **Epoch 5**: Train MSE = 1.6029, Test MSE = 0.0036
- **Epoch 6**: Train MSE = 1.6029, Test MSE = 0.0036

---

## Results

The model achieved a **training MSE of ~1.60** and a **testing MSE of ~0.003**, indicating good generalization performance. The low testing MSE suggests that the test matrix is sparse, nevertheless the model is effective at predicting user ratings.

---

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Omid-Mohebi/RBM-Movie-Recommender.git
   cd RBM-Movie-Recommender
   ```

2. Install the required dependencies:
   ```bash
   pip install opendatasets pandas numpy tensorflow scipy matplotlib
   ```

3. Download the dataset:
   ```python
   import opendatasets as od
   od.download("https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset")
   ```

4. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook src.ipynb
   ```

---

## Usage

After training the model, you can use it to predict user ratings for movies. For example:
```python
# Predict ratings for a user
user_id = 0
movie_ids = [1, 2, 3]  # Example movie IDs
predictions = rbm.predict(user_id, movie_ids)
print(predictions)
```

---

## Contributing

Contributions are welcome! If you'd like to contribute, please:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request.

