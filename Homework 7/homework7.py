############################################################
# CIS 521: Homework 7
############################################################

student_name = "Shubhankar Patankar"

############################################################
# Imports
############################################################

import homework7_data as data

# Include your imports here, if any are used.
from collections import defaultdict

############################################################
# Section 1: Perceptrons
############################################################

class BinaryPerceptron(object):

    def __init__(self, examples, iterations):
        weights = defaultdict(lambda:0) # weights initialized to zero
        for iters in range(iterations):
            for (data_point, true_label) in examples:
                dot_product = sum(weights[key]*data_point.get(key, 0) for key in weights)
                if dot_product > 0:
                    pred_label = True
                else:
                    pred_label = False
                if pred_label != true_label:
                    if true_label == True:
                        weights = {key: weights.get(key, 0) + data_point.get(key, 0)
                                  for key in set(weights) | set(data_point)}
                    else:
                        weights = {key: weights.get(key, 0) - data_point.get(key, 0)
                                  for key in set(weights) | set(data_point)}
        self.weights = weights

    def predict(self, x):
        weights = self.weights
        dot_product = sum(weights[key]*x.get(key, 0) for key in weights)
        if dot_product > 0:
            pred_label = True
        else:
            pred_label = False
        return pred_label

class MulticlassPerceptron(object):

    def __init__(self, examples, iterations):
        weight_dictionaries = defaultdict(lambda: defaultdict(lambda: 0)) # dictionary of dictionaries
        # first level corresponds to a label, second to its associated weights
        labels = []
        for example in examples:
            labels.append(example[1])
        labels = set(labels) # all possible labels in the data
        self.labels = labels 
        for iters in range(iterations):
            for (data_point, true_label) in examples:
                dot_products = defaultdict(lambda: 0) # initialize dot products to take argmax over
                for label in labels:
                    weights = weight_dictionaries[label] # get the weight vector for this label
                    dot_products[label] = sum(weights[key]*data_point.get(key, 0) for key in weights)
                pred_label = max(dot_products, key = dot_products.get) # find label that produces largest dot product
                if pred_label != true_label:
                    true_label_weights = weight_dictionaries[true_label]
                    weight_dictionaries[true_label] = {key: true_label_weights.get(key, 0) + data_point.get(key, 0)
                                              for key in set(true_label_weights) | set(data_point)}
                    # increase weight for the correct label 
                    pred_label_weights = weight_dictionaries[pred_label]
                    weight_dictionaries[pred_label] = {key: pred_label_weights.get(key, 0) - data_point.get(key, 0)
                                              for key in set(pred_label_weights) | set(data_point)}
                    # decrease weight for the incorrect label
        self.weights = weight_dictionaries
        
    def predict(self, x):
        labels = self.labels
        weight_dictionaries = self.weights
        dot_products = defaultdict(lambda: 0)
        for label in labels:
            weights = weight_dictionaries[label]
            dot_products[label] = sum(weights[key]*x.get(key, 0) for key in weights)
        pred_label = max(dot_products, key = dot_products.get)
        return pred_label

############################################################
# Section 2: Applications
############################################################

class IrisClassifier(object):

    def __init__(self, data):
        iterations = 10
        train = []
        for example in data:
            features = example[0]
            label = example[1]
            feature_dict = {}
            for idx, feature in enumerate(features):
                feature_dict[idx] = feature
            train.append((feature_dict, label))
        self.p = MulticlassPerceptron(train, iterations)

    def classify(self, instance):
        p = self.p
        feature_dict = {}
        for idx, feature in enumerate(instance):
            feature_dict[idx] = feature
        return p.predict(feature_dict)

class DigitClassifier(object):

    def __init__(self, data):
        iterations = 10
        train = []
        for (features, label) in data:
            feature_dict = {}
            for idx, feature in enumerate(features):
                feature_dict[idx] = feature
            train.append((feature_dict, label))
        self.p = MulticlassPerceptron(train, iterations)

    def classify(self, instance):
        p = self.p
        feature_dict = {}
        for idx, feature in enumerate(instance):
            feature_dict[idx] = feature
        return p.predict(feature_dict)

class BiasClassifier(object):

    def __init__(self, data):
        iterations = 10
        train = []
        for (feature, label) in data:
            feature_dict = {}
            feature_dict['feature'] = feature
            feature_dict['bias'] = 1
            train.append((feature_dict, label))
        self.p = BinaryPerceptron(train, iterations)

    def classify(self, instance):
        p = self.p
        feature_dict = {}
        feature_dict['feature'] = instance
        feature_dict['bias'] = 1
        return p.predict(feature_dict)

class MysteryClassifier1(object):

    def __init__(self, data):
        iterations = 15
        train = []
        for (features, label) in data:
            feature_dict = {0: features[0], 1: features[1], 2: features[0]**2 + features[1]**2, 3: 1}
            train.append((feature_dict, label))
        self.p = MulticlassPerceptron(train, iterations)

    def classify(self, instance):
        p = self.p
        feature_dict = {0: instance[0], 1: instance[1], 2: instance[0]**2 + instance[1]**2, 3: 1}
        return p.predict(feature_dict)

class MysteryClassifier2(object):

    def __init__(self, data):
        iterations = 10
        train = []
        for (features, label) in data:
            feature_dict = {0: features[0] * features[1] * features[2]}
            train.append((feature_dict, label))
        self.p = MulticlassPerceptron(train, iterations)

    def classify(self, instance):
        p = self.p
        feature_dict = {0: instance[0] * instance[1] * instance[2]}
        return p.predict(feature_dict)

############################################################
# Section 3: Feedback
############################################################

feedback_question_1 = 10

feedback_question_2 = """
Figuring out which features to append in the mystery datasets was challenging.
It would have been nice to see examples in class of how to visualize the datasets, and how to judge the best choice of extra features.
"""

feedback_question_3 = """
I liked that we got to implement the perceptrons from scratch. 
I would add a brief demonstration on how to visualize the high dimensional datasets.
"""
