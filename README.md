# Spam-Email-Classifier
Spam Email predictor project for Codec Technologies. Use techniques like Naive Bayes or Support Vector Machines (SVM) for classification.
# Spam Email Classifier

This repository contains a machine learning model to classify emails as spam or non-spam (ham). The model is built using TensorFlow and Keras, utilizing an LSTM (Long Short-Term Memory) network to process the text data.

## Dataset

The dataset used in this project is `spam_ham_dataset.csv`, which contains a collection of emails labeled as either 'spam' or 'ham'.

## Project Steps

The project follows these key steps:

1.  **Import Required Libraries:** Import necessary libraries for data manipulation, visualization, text processing, and model building.
2.  **Load and Explore Data:** Load the dataset and perform initial exploration to understand its structure and content.
3.  **Visualize Label Distribution:** Analyze the distribution of spam and ham emails in the dataset.
4.  **Balance the Dataset:** Address the class imbalance by downsampling the majority class (ham emails) to match the number of spam emails.
5.  **Clean the Text:**
    *   Remove "Subject:" prefix from the email text.
    *   Remove punctuation from the email text.
    *   Remove stop words from the email text.
6.  **Visualize Word Cloud:** Generate word clouds for both spam and non-spam emails to visualize the most frequent words in each category.
7.  **Tokenization and Padding:**
    *   Split the data into training and testing sets.
    *   Tokenize the email text to convert words into sequences of integers.
    *   Pad the sequences to a fixed length.
8.  **Define the Model:** Build a sequential model using TensorFlow/Keras, including an Embedding layer, LSTM layer, and Dense layers.
9.  **Compile the Model:** Configure the model with a loss function, optimizer, and metrics.
10. **Train the Model:** Train the model on the training data, using Early Stopping and ReduceLROnPlateau callbacks for better training performance.
11. **Evaluate the Model:** Evaluate the trained model on the test data to measure its performance (loss and accuracy).
12. **Visualize Training History:** Plot the training and validation accuracy over epochs to visualize the model's learning progress.

## Code

The Python code for this project is available in the Jupyter Notebook file in this repository.

## Setup and Usage

1.  Clone this repository.
2.  Make sure you have the necessary libraries installed (see the imports in the notebook). You can install them using pip: `pip install pandas numpy matplotlib seaborn nltk tensorflow scikit-learn wordcloud`
3.  Download the `spam_ham_dataset.csv` file and place it in the same directory as the notebook.
4.  Run the Jupyter Notebook cells sequentially to execute the code and train the model.

## Results

The model achieved a test accuracy of approximately 98%, demonstrating its effectiveness in classifying spam emails.

## Further Improvements

*   Experiment with different model architectures (e.g., adding more layers, different types of layers).
*   Tune hyperparameters for better performance.
*   Explore other text preprocessing techniques.
*   Consider using a larger dataset for training.

## Author

[AMAAN SAYED /sayed-Amaan6104]

## License

[Choose a license and add details here]
