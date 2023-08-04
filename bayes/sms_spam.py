import math
from collections import defaultdict

# Step 1: Load the dataset
def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            label, text = line.strip().split('\t')
            dataset.append((text, label))
    return dataset

# Step 2: Split the dataset into training and testing sets
def split_dataset(dataset, split_ratio=0.8):
    split_index = int(len(dataset) * split_ratio)
    train_set = dataset[:split_index]
    test_set = dataset[split_index:]
    return train_set, test_set

# Step 3: Train the Naive Bayes classifier
def train_naive_bayes(train_set):
    # Count the occurrences of each word and the total words in each class
    class_word_counts = defaultdict(lambda: defaultdict(int))
    class_total_words = defaultdict(int)
    vocabulary = set()

    for text, label in train_set:
        words = text.lower().split()
        class_word_counts[label]['__total__'] += len(words)
        class_total_words[label] += len(words)

        for word in words:
            class_word_counts[label][word] += 1
            vocabulary.add(word)

    # Calculate the prior probability of each class
    total_docs = len(train_set)
    class_priors = {label: math.log(count / total_docs) for label, count in class_total_words.items()}

    # Calculate the conditional probabilities
    class_cond_probs = defaultdict(lambda: defaultdict(float))
    for label in class_word_counts:
        for word in vocabulary:
            word_count = class_word_counts[label][word]
            total_words_in_class = class_word_counts[label]['__total__']
            class_cond_probs[label][word] = math.log((word_count + 1) / (total_words_in_class + len(vocabulary)))

    return class_priors, class_cond_probs, vocabulary

# Step 4: Predict the class of a test instance
def predict_naive_bayes(text, class_priors, class_cond_probs, vocabulary):
    words = text.lower().split()

    scores = {}
    for label in class_priors:
        score = class_priors[label]
        for word in words:
            if word in vocabulary:
                score += class_cond_probs[label][word]
        scores[label] = score

    predicted_label = max(scores, key=scores.get)
    return predicted_label

# Step 5: Evaluate the classifier using the test set
def evaluate_naive_bayes(test_set, class_priors, class_cond_probs, vocabulary):
    correct_predictions = 0

    for text, label in test_set:
        predicted_label = predict_naive_bayes(text, class_priors, class_cond_probs, vocabulary)
        if predicted_label == label:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_set)
    return accuracy

# Load the dataset
dataset = load_dataset('sms_spam_collection.txt')

# Split the dataset into training and testing sets
train_set, test_set = split_dataset(dataset, split_ratio=0.8)

# Train the Naive Bayes classifier
class_priors, class_cond_probs, vocabulary = train_naive_bayes(train_set)

# Evaluate the classifier
accuracy = evaluate_naive_bayes(test_set, class_priors, class_cond_probs, vocabulary)
print(f"Accuracy: {accuracy:.2%}")
