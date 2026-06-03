import numpy as np
from src.imbalanced import util
from random import random
import os

### NOTE : You need to complete logreg implementation first!
from sklearn.linear_model import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def get_accuracy_score(y_valid_input, y_predict_input):
    accuracy_count = 0
    for index, _ in enumerate(y_valid_input):
        if y_predict_input[index] == y_valid_input[index]:
            accuracy_count += 1
    return accuracy_count / len(y_valid_input)

def get_positive_accuracy(y_valid_input, y_predict_input):
    positive_accuracy_count = 0
    positive_count = 0
    for index, _ in enumerate(y_valid_input):
        if y_valid_input[index] == 1.0:
            positive_count +=1
            if y_predict_input[index] == 1.0:
                positive_accuracy_count += 1
    return float(positive_accuracy_count) / float(positive_count)

def get_negative_accuracy(y_valid_input, y_predict_input):
    negative_accuracy_count = 0
    negative_count = 0
    for index, _ in enumerate(y_valid_input):
        if y_valid_input[index] == 0.0:
            negative_count +=1
            if y_predict_input[index] == 0.0:
                negative_accuracy_count += 1
    return float(negative_accuracy_count) / float(negative_count)

def get_balanced_accuracy_score(y_valid_input, y_predict_input):
    a_zero = get_negative_accuracy(y_valid_input, y_predict_input)
    a_one = get_positive_accuracy(y_valid_input, y_predict_input)
    return (1/2) * (a_zero + a_one)

def produce_sampled_validation_file(validation_path, sampled_validation_path):
    """
    Given an input validation file, produces a sampled validation file
    such that all negative examples of source validation file appears as is, 
    all positive samples appear with a repitition factor of 1/kappa

    :param validation_path: Source validation file from which we should generate a sampled validation file
    :param sampled_validat_path: Target sampled validation file that is extracted from the source file
    """
    positive_list=[]
    negative_list =[]
    header = None
    try:
        with open(validation_path, 'r', encoding='utf-8') as file:
            for line in file:
                cleaned_line = line.strip()
                if header is None:
                    header = cleaned_line
                    continue
                splits = cleaned_line.split(",")
                if splits[2] == '0.0':
                    negative_list.append(cleaned_line)
                if splits[2] == '1.0':
                    positive_list.append(cleaned_line)        
    except FileNotFoundError:
        print(f"Error: The validation input file '{validation_path}' was not found.")
    except IOError as e:
        print(f"Error: An I/O error occurred - {e}")

    majority_list = None
    minority_list = None
    if len(positive_list) < len(negative_list):
        minority_list = positive_list
        majority_list = negative_list
    else:
        minority_list = negative_list
        majority_list = positive_list
   
    sampled_validation_path = os.path.join(script_dir, 'validation_sampled.csv')
    with open(sampled_validation_path, 'w',  encoding='utf-8') as file:
        file.write(header + '\n')
        for line in majority_list:
            file.write(line + '\n')
        for line in minority_list:
            for i in range (0, int(1/kappa)):
                file.write(line+'\n')

def main(train_path, validation_path, save_path):
    """Problem: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt() as a 1D numpy array
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)

    x_valid, y_valid = util.load_dataset(validation_path, add_intercept=True)
    y_predict = log_reg.predict(x_valid)
    np.savetxt(output_path_naive, y_predict)

    accuracy = get_accuracy_score(y_valid, y_predict)
    print("accuracy={}".format(accuracy))
    balanced_accuracy = get_balanced_accuracy_score(y_valid, y_predict)
    print("balanced_accuracy={}".format(balanced_accuracy))
    accuracy_a_zero = get_negative_accuracy(y_valid, y_predict)
    print("accuracy_a_zero (negative)={}".format(accuracy_a_zero))
    accuracy_a_one = get_positive_accuracy(y_valid, y_predict)
    print("accuracy_a_one (positive)={}".format(accuracy_a_one))
    util.plot(x_valid, y_valid, log_reg.coef_[0], os.path.join(script_dir, "validation_plot_naive.png")) 
    
    print("\n\n")
    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt() as a 1D numpy array
    # Repeat minority examples 1 / kappa times
    # *** END CODE HERE

    sampled_validation_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validation_sampled.csv')
    # does the sampling with 1/k for minority examples and writes to sampled_validation_file_path
    produce_sampled_validation_file(validation_path, sampled_validation_file_path)

    x_valid_sampled, y_valid_sampled = util.load_dataset(sampled_validation_file_path, add_intercept=True)
    y_predict_sampled = log_reg.predict(x_valid_sampled)
    np.savetxt(output_path_upsampling, y_predict_sampled)
    
    accuracy_sampled = get_accuracy_score(y_valid_sampled, y_predict_sampled)
    print("accuracy_sampled={}".format(accuracy_sampled))
    balanced_accuracy_sampled = get_balanced_accuracy_score(y_valid_sampled, y_predict_sampled)
    print("balanced_accuracy_sampled={}".format(balanced_accuracy_sampled))
    accuracy_a_zero_sampled = get_negative_accuracy(y_valid_sampled, y_predict_sampled)
    print("accuracy_a_zero_sampled (negative)={}".format(accuracy_a_zero_sampled))
    accuracy_a_one_sampled = get_positive_accuracy(y_valid_sampled, y_predict_sampled)
    print("accuracy_a_one_sampled (positive)={}".format(accuracy_a_one_sampled))
    util.plot(x_valid_sampled, y_valid_sampled, log_reg.coef_[0], os.path.join(script_dir, "validation_plot_sampled.png")) 
    

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main(train_path=os.path.join(script_dir, 'train.csv'),
         validation_path=os.path.join(script_dir, 'validation.csv'),
         save_path=os.path.join(script_dir, 'imbalanced_X_pred.txt'))
