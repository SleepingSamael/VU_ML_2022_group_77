
#load in packages
import os
import time
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

from sklearn.metrics import roc_curve, auc



random_state = 7

# Reading the data
df = pd.read_csv("./diabetes/training_data(no_pre-diabetes).csv")
# select Diabetes_binary as target variable:
y = df['Diabetes_012']
y = y.replace(2.0, 1)
# select all the other columns minus Diabetes_binary as the feature variables:
X = df.drop(['Diabetes_012'], axis=1)

test_data = pd.read_csv("./diabetes/testing_data(no_pre-diabetes).csv")
x_test = test_data.iloc[:, 1:]
y_test = test_data.Diabetes_012
y_test = y_test.replace(2.0, 1)


def standardization(x, x_test):
    # numerical features
    numeric_features = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
    # copy of datasets
    X_train_stand = x.copy()
    X_test_stand = x_test.copy()
    # apply standardization on numerical features
    try:
        for i in numeric_features:
            # fit on training data column
            scale = StandardScaler().fit(X_train_stand[[i]])
            # transform the training data column
            X_train_stand[i] = scale.transform(X_train_stand[[i]])
            # transform the testing data column
            X_test_stand[i] = scale.transform(X_test_stand[[i]])
    except Exception as e:
        pass

    X = X_train_stand
    x_test = X_test_stand
    return X, x_test


def normalization(x, x_test):
    # data normalization with sklearn
    # fit scaler on training data
    norm = MinMaxScaler().fit(x)
    # transform training data
    X = norm.transform(x)
    # transform testing dataabs
    x_test = norm.transform(x_test)
    return X, x_test


def over_sample(x, y):
    # oversample
    ros = RandomOverSampler()
    X, y = ros.fit_resample(x, y)
    return X, y


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

x_stand_train, x_stand_test = standardization(X, x_test)
x_stand_train, x_stand_test = normalization(x_stand_train, x_stand_test)
#x_stand_train, y_stand_train = over_sample(x_stand_train, y)


def train_with_cross_validate(x_stand_train,y_stand_train):
    # cross_validate method
    classifier_name = 'Simple Neural Network: MLPClassifier'
    start_ts = time.time()
    # try swapping out the classifier for a different one or changing the parameters
    clf = MLPClassifier(activation='relu', solver='adam', alpha=0.0001, max_iter=1000, hidden_layer_sizes=(100,),
                        random_state=random_state)

    from functools import partial
    from sklearn.metrics import precision_score, make_scorer

    scores = cross_validate(clf, x_stand_train, y_stand_train, scoring=scorers, cv=10, return_train_score=True)

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


def train_with_feature_selection():
    # Model Building with feature selection
    selected_feat = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
    X_feat = X[selected_feat]
    y_feat = y
    x_test_feat = x_test[selected_feat]
    x_stand_train, x_stand_test = standardization(X_feat, x_test_feat)
    x_stand_train, x_stand_test = normalization(x_stand_train, x_stand_test)
    #x_stand_train, y_stand_train = over_sample(x_stand_train, y_feat)

    classifier_name = 'Simple Neural Network: MLPClassifier w/ Feature Selection:'

    start_ts = time.time()
    # Changed the X to X_feat and y to y_feat
    clf = MLPClassifier(activation='relu', solver='adam', alpha=0.0001, max_iter=1000, hidden_layer_sizes=(100,),
                        random_state=random_state)
    scores = cross_validate(clf, x_stand_train, y, scoring=scorers, cv=10)

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


def train_with_GridSearchCV(x_stand_train,y_stand_train):

    parameters = {'solver': ['adam'], 'max_iter': [2000],
                  'alpha': [0.0001], 'hidden_layer_sizes': np.arange(10, 20),
                  'random_state': [7]}
    clf = MLPClassifier()

    #clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=3)
    clf.fit(x_stand_train, y_stand_train)
    return clf


def get_output(clf):
    test_pred = clf.predict(x_stand_test)
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

    fpr, tpr, threshold = roc_curve(y_test, clf.predict_proba(x_stand_test))
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

def start():
    model = train_with_GridSearchCV(x_stand_train,y)
    get_output(model)

if __name__ == '__main__':
    train_with_feature_selection()