import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])

    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    
import csv

def load_data(filename): 
    evidence = []
    labels = []
    month_to_index = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
                      "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}
    visitor_type_to_index = {"Returning_Visitor": 1, "New_Visitor": 0, "Other": 0}
    string_to_bool = {"TRUE": 1, "FALSE": 0}
    
    with open(filename, 'r') as file:
        cr = csv.reader(file, delimiter=",")
        next(cr)
        for row in cr:
            row[0] = int(row[0])  # Administrative
            row[1] = float(row[1])  # Administrative_Duration
            row[2] = int(row[2])  # Informational
            row[3] = float(row[3])  # Informational_Duration
            row[4] = int(row[4])  # ProductRelated
            row[5] = float(row[5])  # ProductRelated_Duration
            row[6] = float(row[6])  # BounceRates
            row[7] = float(row[7])  # ExitRates
            row[8] = float(row[8])  # PageValues
            row[9] = float(row[9])  # SpecialDay
            row[10] = month_to_index[row[10]]  # Month
            row[11] = int(row[11])  # OperatingSystems
            row[12] = int(row[12])  # Browser
            row[13] = int(row[13])  # Region
            row[14] = int(row[14])  # TrafficType
            row[15] = visitor_type_to_index[row[15]]  # VisitorType
            row[16] = string_to_bool[row[16]] # Weekend
            evidence.append(row[:-1])
            labels.append(string_to_bool[row[-1]])  # Revenue

    return (evidence, labels)

    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
