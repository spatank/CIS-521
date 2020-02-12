############################################################
# CIS 521: Neural Network for Fashion MNIST Dataset
############################################################

student_name = "Shubhankar Patankar"

############################################################
# Imports
############################################################

import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools

# Include your imports here, if any are used.
import numpy as np
import csv

############################################################
# Neural Networks
############################################################

def load_data(file_path, reshape_images):
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        fields = next(csv_reader)
        data = [data for data in csv_reader]
        data_array = np.asarray(data, dtype = int)
        images = data_array[:, 1:]
        labels = data_array[:, 0]
        if reshape_images:
            images = images.reshape((20000, 1, 28, 28))
    return images, labels

# def load_data(file_path, reshape_images):
#     with open(file_path) as csv_file:
#         csv_reader = csv.reader(csv_file)
#         fields = next(csv_reader)
#         data = np.array(list(csv_reader)).astype(float)
#         images = data[:, 1:]
#         labels = data[:, 0]
#         if reshape_images:
#             images = images.reshape((20000, 1, 28, 28))
#     return images, labels

# def load_data(file_path, reshape_images):
#     dataset = pd.read_csv(file_path)
#     fields = list(dataset.columns.values)
#     all_images = dataset[fields[1:]]
#     all_labels = dataset[fields[0]]
#     all_images = np.array(all_images)
#     all_labels = np.array(all_labels)
#     if reshape_images:
#         all_images = all_images.reshape((20000, 1, 28, 28))
#     return all_images, all_labels


# PART 2.2
class EasyModel(torch.nn.Module):
    def __init__(self):
        super(EasyModel, self).__init__()
        self.fc = torch.nn.Linear(784, 10) # fc --> fully connected

    def forward(self, x):
        return self.fc(x)


# PART 2.3
class MediumModel(torch.nn.Module):
    def __init__(self):
        super(MediumModel, self).__init__()
        self.fc1 = torch.nn.Linear(784, 200)
        self.fc2 = torch.nn.Linear(200, 200)
        self.fc3 = torch.nn.Linear(200, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x)


# PART 2.4
class AdvancedModel(torch.nn.Module):
    def __init__(self):
        super (AdvancedModel,self).__init__()
        self.cnn1 = torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 2)
        self.relu1 = torch.nn.ELU()
        torch.nn.init.xavier_uniform(self.cnn1.weight)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size = 2)
        self.cnn2 = torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 2)
        self.relu2 = torch.nn.ELU()
        torch.nn.init.xavier_uniform(self.cnn2.weight)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size = 2)
        self.fcl = torch.nn.Linear(32*7*7, 10)
        
    def forward(self,x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0),-1)
        out = self.fcl(out)
        return out


############################################################
# Fashion MNIST dataset
############################################################

class FashionMNISTDataset(Dataset):
    def __init__(self, file_path, reshape_images):
        self.X, self.Y = load_data(file_path, reshape_images)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

############################################################
# Reference Code
############################################################

def train(model, data_loader, num_epochs, learning_rate):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data_loader):
            images = torch.autograd.Variable(images.float())
            labels = torch.autograd.Variable(labels)

            optimizer.zero_grad()
            outputs = model(images.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                y_true, y_predicted = evaluate(model, data_loader)
                print(f'Epoch : {epoch}/{num_epochs}, '
                      f'Iteration : {i}/{len(data_loader)},  '
                      f'Loss: {loss.item():.4f},',
                      f'Train Accuracy: {100.* accuracy_score(y_true, y_predicted):.4f},',
                      f'Train F1 Score: {100.* f1_score(y_true, y_predicted, average="weighted"):.4f}')


def evaluate(model, data_loader):
    model.eval()
    y_true = []
    y_predicted = []
    for images, labels in data_loader:
        images = torch.autograd.Variable(images.float())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels)
        y_predicted.extend(predicted)
    return y_true, y_predicted


def plot_confusion_matrix(cm, class_names, title=None):
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def main():
    class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    num_epochs = 2
    batch_size = 100
    learning_rate = 0.001
    file_path = 'dataset.csv'

    data_loader = torch.utils.data.DataLoader(dataset=FashionMNISTDataset(file_path, False),
                                              batch_size=batch_size,
                                              shuffle=True)
    data_loader_reshaped = torch.utils.data.DataLoader(dataset=FashionMNISTDataset(file_path, True),
                                                       batch_size=batch_size,
                                                       shuffle=True)

    # EASY MODEL
    easy_model = EasyModel()
    train(easy_model, data_loader, num_epochs, learning_rate)
    y_true_easy, y_pred_easy = evaluate(easy_model, data_loader)
    print(f'Easy Model: '
          f'Final Train Accuracy: {100.* accuracy_score(y_true_easy, y_pred_easy):.4f},',
          f'Final Train F1 Score: {100.* f1_score(y_true_easy, y_pred_easy, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_easy, y_pred_easy), class_names, 'Easy Model')

    # MEDIUM MODEL
    medium_model = MediumModel()
    train(medium_model, data_loader, num_epochs, learning_rate)
    y_true_medium, y_pred_medium = evaluate(medium_model, data_loader)
    print(f'Medium Model: '
          f'Final Train Accuracy: {100.* accuracy_score(y_true_medium, y_pred_medium):.4f},',
          f'Final F1 Score: {100.* f1_score(y_true_medium, y_pred_medium, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_medium, y_pred_medium), class_names, 'Medium Model')

    # ADVANCED MODEL
    advanced_model = AdvancedModel()
    train(advanced_model, data_loader_reshaped, num_epochs, learning_rate)
    y_true_advanced, y_pred_advanced = evaluate(advanced_model, data_loader_reshaped)
    print(f'Advanced Model: '
          f'Final Train Accuracy: {100.* accuracy_score(y_true_advanced, y_pred_advanced):.4f},',
          f'Final F1 Score: {100.* f1_score(y_true_advanced, y_pred_advanced, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_advanced, y_pred_advanced), class_names, 'Advanced Model')

############################################################
# Feedback
############################################################

feedback_question_1 = """
The EasyModel() was often confused between coats and shirts, and between pullovers and coats.
"""

feedback_question_2 = """
The advanced model has two convolutional layers with ReLU activations. This is followed by max pooling.
The last layer is a fully connected layer.
"""

feedback_question_3 = 20

feedback_question_4 = """
The first section of the assignment on coding functions for CNNs was unnecessary in my view. 
It did offer the means to really understand what convolution and pooling do. 
However, it detracted from being able to learn more about PyTorch. 
"""

feedback_question_5 = """
The PyTorch section on implementing NNs was very useful.
It introduced me to a deep learning framework that I have been meaning to learn but avoiding out of inertia.
I would have preferred for this section to have had a more gradual build-up like some of the earlier course assignments.
It would have been nice to implement an advanced model piece-by-piece just the way we did with smaller methods implementations for classes.
"""

if __name__ == '__main__':
    main()
