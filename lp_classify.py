import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import average_precision_score
import sys
from tqdm import tqdm
from time_check import tic
from time_check import toc

alpha = sys.argv[1]
edge_features = np.load('./edges/edge_features'+str(alpha)+'.npy')
edge_labels = np.load('./edges/edge_labels'+str(alpha)+'.npy')
training_size = int(0.8*edge_labels.shape[0])
train_x = edge_features[:training_size]
test_x = edge_features[training_size:]
train_y = edge_labels[:training_size]
test_y = edge_labels[training_size:]


def training_SVM(train_x, train_y):
    model = SVC(gamma='auto')
    print('Traning SVM...')
    tic()
    model.fit(train_x, train_y)
    toc()
    return model

def training_LR(train_x, train_y):
    model = LR()
    print('Traning Linear Regression...')
    # tic()
    model.fit(train_x, train_y)
    # toc()
    return model

# model = training_SVM(train_x, train_y)
model = training_LR(train_x, train_y)

acc = model.score(test_x, test_y)
print('Testing Acc:', acc)
preds = model.decision_function(test_x)
AP = average_precision_score(test_y, preds)
# AP = np.mean(precisions)
print('Testing AP:', AP)
# print('Testing AP:', AP)
