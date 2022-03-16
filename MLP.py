
#load in packages
import os
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier


random_state = 7

# Reading the data
df = pd.read_csv("./diabetes/training_data(no_pre-diabetes).csv")

# Model building with no feature selection

# select Diabetes_binary as target variable:
y = df['Diabetes_012']
y = y.replace(2.0, 1)

# select all the other columns minus Diabetes_binary as the feature variables:
X = df.drop(['Diabetes_012'], axis=1)


# create true negative, false positive, false negative, and true positive
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]


#Setup classifier scorers
scorers = {'Accuracy': 'accuracy',
           'roc_auc': 'roc_auc',
           'Sensitivity': 'recall',
           'precision': 'precision',
           'tp': make_scorer(tp),
           'tn': make_scorer(tn),
           'fp': make_scorer(fp),
           'fn': make_scorer(fn)}

def start_with_cross_validate():
    # cross_validate method
    classifier_name = 'Simple Neural Network: MLPClassifier'
    start_ts = time.time()
    # try swapping out the classifier for a different one or changing the parameters
    clf = MLPClassifier(activation='logistic', alpha=0.0001, max_iter=1000, hidden_layer_sizes=(10,),
                        random_state=random_state)

    from functools import partial
    from sklearn.metrics import precision_score, make_scorer

    scores = cross_validate(clf, X, y, scoring=scorers, cv=5, return_train_score=True)

    Sensitivity = round(scores['test_tp'].mean() / (scores['test_tp'].mean() + scores['test_fn'].mean()),
                        3) * 100  # TP/(TP+FN) also recall
    Specificity = round(scores['test_tn'].mean() / (scores['test_tn'].mean() + scores['test_fp'].mean()),
                        3) * 100  # TN/(TN+FP)
    PPV = round(scores['test_tp'].mean() / (scores['test_tp'].mean() + scores['test_fp'].mean()),
                3) * 100  # PPV = tp/(tp+fp) also precision
    NPV = round(scores['test_tn'].mean() / (scores['test_fn'].mean() + scores['test_tn'].mean()), 3) * 100  # TN(FN+TN)

    scores_Acc = scores['test_Accuracy']
    print(f"{classifier_name} Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))
    scores_AUC = scores['test_roc_auc']  # Only works with binary classes, not multiclass
    print(f"{classifier_name} AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))
    scores_sensitivity = scores['test_Sensitivity']  # Only works with binary classes, not multiclass
    print(f"{classifier_name} Recall: %0.2f (+/- %0.2f)" % (scores_sensitivity.mean(), scores_sensitivity.std() * 2))
    scores_precision = scores['test_precision']  # Only works with binary classes, not multiclass
    print(f"{classifier_name} Precision: %0.2f (+/- %0.2f)" % (scores_precision.mean(), scores_precision.std() * 2))
    print(f"{classifier_name} Sensitivity = ", Sensitivity, "%")
    print(f"{classifier_name} Specificity = ", Specificity, "%")
    print(f"{classifier_name} PPV = ", PPV, "%")
    print(f"{classifier_name} NPV = ", NPV, "%")

    print(scores['test_tp'].mean(),scores['test_tn'].mean(), scores['test_fp'].mean(),scores['test_fn'].mean())

    print("Runtime:", time.time() - start_ts)

    # Model Building with feature selection
    # select Diabetes_binary as target variable:
    y_feat = df['Diabetes_012']
    y_feat = y_feat.replace(2.0, 1)

    # select all the other columns minus Diabetes_binary as the feature variables:
    X_feat = df.drop(['Diabetes_012'], axis=1)
    X_feat = X_feat[["HighBP", "HighChol", "BMI", "HeartDiseaseorAttack", "GenHlth", "PhysHlth", "DiffWalk", "Age"]]

    classifier_name = 'Simple Neural Network: MLPClassifier w/ Feature Selection:'

    start_ts = time.time()
    # Changed the X to X_feat and y to y_feat
    clf = MLPClassifier(activation='logistic', solver='adam', alpha=0.0001, max_iter=1000, hidden_layer_sizes=(10,),
                        random_state=random_state)
    scores = cross_validate(clf, X_feat, y_feat, scoring=scorers, cv=5)

    Sensitivity = round(scores['test_tp'].mean() / (scores['test_tp'].mean() + scores['test_fn'].mean()),
                        3) * 100  # TP/(TP+FN) also recall
    Specificity = round(scores['test_tn'].mean() / (scores['test_tn'].mean() + scores['test_fp'].mean()),
                        3) * 100  # TN/(TN+FP)
    PPV = round(scores['test_tp'].mean() / (scores['test_tp'].mean() + scores['test_fp'].mean()),
                3) * 100  # PPV = tp/(tp+fp) also precision
    NPV = round(scores['test_tn'].mean() / (scores['test_fn'].mean() + scores['test_tn'].mean()), 3) * 100  # TN(FN+TN)

    scores_Acc = scores['test_Accuracy']
    print(f"{classifier_name} Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))
    scores_AUC = scores['test_roc_auc']  # Only works with binary classes, not multiclass
    print(f"{classifier_name} AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))
    scores_sensitivity = scores['test_Sensitivity']  # Only works with binary classes, not multiclass
    print(f"{classifier_name} Recall: %0.2f (+/- %0.2f)" % (scores_sensitivity.mean(), scores_sensitivity.std() * 2))
    scores_precision = scores['test_precision']  # Only works with binary classes, not multiclass
    print(f"{classifier_name} Precision: %0.2f (+/- %0.2f)" % (scores_precision.mean(), scores_precision.std() * 2))
    print(f"{classifier_name} Sensitivity = ", Sensitivity, "%")
    print(f"{classifier_name} Specificity = ", Specificity, "%")
    print(f"{classifier_name} PPV = ", PPV, "%")
    print(f"{classifier_name} NPV = ", NPV, "%")

    print("Runtime:", time.time() - start_ts)

def start_without_cross_validate():
    clf = MLPClassifier(activation='logistic', alpha=0.0001, max_iter=1000, hidden_layer_sizes=(10,),
                        random_state=random_state)
    clf.fit(X,y)
    test_data = pd.read_csv("./diabetes/testing_data(no_pre-diabetes).csv")
    x_test = test_data.iloc[:, 1:]
    y_test = test_data.Diabetes_012
    y_test = y_test.replace(2.0, 1)

    test_pred = clf.predict(x_test)
    # Accuracy
    confusion_hard = confusion_matrix(y_test, test_pred)
    accuracy = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    print('\nMLP Accuracy for validation set=: {0:.4f}, \nprecision: {1:.4f}, \nrecall: {2:.4f},\
    \nF1: {3:.4f}'.format(accuracy, precision, recall, f1))

    # confusion_matrix plot
    import seaborn as sns
    class_label = ["No_Diabetes", "Diabetes"]
    df_cm = pd.DataFrame(confusion_hard, index=class_label, columns=class_label)
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion matrix for mlp', fontsize=20) # title with fontsize 20
    plt.savefig("cm.png", dpi=300)


if __name__ == '__main__':
    start_with_cross_validate()