import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, log_loss, f1_score
from sklearn.metrics import plot_confusion_matrix, classification_report, precision_recall_curve, auc
from sklearn.utils import resample
from sklearn.decomposition import PCA
from xgboost import XGBClassifier,  plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pickle
import time
import seaborn as sns


def generate_offer_data(data, offer_id):
    """
    Generate the train, test data for each offer id
    
    INPUT:
    - data: DataFrame
    - offer_id: int
    
    OUTPUT:
    X_train, DataFrame,  training features
    X_test,  DataFrame,  testing features
    Y_train, array, training labels
    Y_test, array, testing labels
    """
    d = data[data['offer_id']==1.0]
    X, y = d.drop(columns=['viewed_and_above_mean']), d['viewed_and_above_mean']

    # scale the features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), 
                             columns = X.columns,index=X.index)
    print('class 0 to class 1 ratio', np.sum(y==0)/np.sum(y==1))

    # split the train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    y_train, y_test = y_train.values.reshape(-1), y_test.values.reshape(-1)
    print('The numbers of class 1 is ', np.sum(y_train), 'numbers of class 0 is',  len(y_train)-np.sum(y_train))
    return X_train, X_test, y_train, y_test


def fit_classifier(clf, X_train, y_train, scoring):

    """
    Fits a classifier to its training data using GridSearchCV and calculates f1_score
    
    INPUT:
    - clf (classifier): classifier to fit
    - param_grid (dict): classifier parameters used with GridSearchCV
    - X(DataFrame): training features
    - y(DataFrame): training label
            
    OUTPUT:
    - classifier: input classifier fitted to the training data
    """
    
    # cv uses StratifiedKFold
    # scoring roc_auc available as parameter
    start = time.time()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    print("Training {} :".format(clf.__class__.__name__))
    scores = cross_val_score(clf, X_train, y_train, scoring=scoring, cv=cv)
    end = time.time()
    time_taken = round(end-start,2)

    print(clf.__class__.__name__)
    print("Time taken : {} secs".format(time_taken))
    print("roc_auc_weighted : {}".format(np.mean(scores)))
    
            
    yhat = cross_val_predict(clf, X_train, y_train, cv=5, method='predict_proba')
    # calculate the precision-recall auc
    pos_probs = yhat[:, 1]
    precision, recall, thres= precision_recall_curve(y_train, pos_probs)
    print('PR AUC:',auc(recall, precision))
    plt.plot(thres, recall[:-1])
    plt.plot(thres, precision[:-1])
    plt.legend(['recall', 'precision'])
    plt.xlabel('threshold')
    plt.show()
    
    
    print("*"*20)
    
    return scores



def try_base_models(X_train, y_train, scoring):
    """
    Try to fit the few base models (here i choose the Logistic Regression model, 
    XGBClassifier, Random Forest Classifier and Adaboost Classifier) . 
    Then, print the cross validated ROC_AUC score, PR_AUC score, and plot the recall vs precision.
    
    INPUT:
    X_train, DataFrame,  training features
    Y_train, array, training labels
    scoring, str,  scoring metric used in the cross validation
    
    OUTPUT:
    none
    """
    lr = LogisticRegression(penalty='l1', solver='saga', max_iter = 5000)
    rfc = RandomForestClassifier(random_state=42) # RandomForestClassifier
    abc = AdaBoostClassifier(random_state=42) # AdaBoostClassifier
    gbc = XGBClassifier(random_state=42)#gradientboost classifier

    result = {}
    print("*"*60)
    print('')
    models = []
    scores = []
    for classifier in [gbc, rfc, abc]:
        score = fit_classifier(classifier, X_train, y_train, scoring)
        models.append(classifier.__class__.__name__) 
        scores.append(score)
           
    result['no_resample'] = {'models': models, 'scores':score}
    
    
    return

def try_over_sample_and_changing_class_weight(model_name, X_train, y_train):
    """
    Check the two methods to deal with imbalanced dataset: over sample the minority 
    class or change the class weight of the model.
    
    INPUT:
    X_train, DataFrame,  training features
    Y_train, array, training labels
    model_name, str,  name of the classifier
    
    OUTPUT:
    none
    """
    if model_name =='XGBClassifier':
        gbc = XGBClassifier(random_state=42)
        print('With over sampling:'+'*'*40)
        over = SMOTE(random_state=42)
        steps = [('over', over), ('model', gbc)]
        pipeline = Pipeline(steps=steps)
        fit_classifier(pipeline, X_train, y_train, 'roc_auc_ovr_weighted')

        print('Changing the class weighting hyperparameters:'+'*'*40)
        # Tune the Class Weighting hyperparameters
        gbc = XGBClassifier(scale_pos_weight=5, random_state=42)
        score = fit_classifier(gbc, X_train, y_train, 'roc_auc_ovr_weighted')
    
    if model_name == 'RandomForestClassifier':
        rfc = RandomForestClassifier(random_state=42)
        print('With over sampling:'+'*'*40)
        over = SMOTE(random_state=42)
        steps = [('over', over), ('model', rfc)]
        pipeline = Pipeline(steps=steps)
        fit_classifier(pipeline, X_train, y_train, 'roc_auc_ovr_weighted')
        
        print('Changing the class weighting hyperparameters:'+'*'*40)
        # Tune the Class Weighting hyperparameters
        gbc = RandomForestClassifier(class_weight={0:1, 1:5}, random_state=42)
        score = fit_classifier(gbc, X_train, y_train, 'roc_auc_ovr_weighted')  
    return



def plot_feature_importance(X_train, y_train):
    """
    Plot the feature importances by fiting a XGBClassifier
    
    INPUT:
    X_train, DataFrame,  training features
    Y_train, array, training labels
    
    OUTPUT:
    none
    
    """
    gbc = XGBClassifier(scale_pos_weight=5, random_state=42)
    gbc.fit(X_train, y_train)
    # plot feature importance
    fig, ax = plt.subplots(figsize=(20, 10))
    ax = plot_importance(gbc, height=0.5, ax=ax, max_num_features= 20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Features importance', fontsize=18)
    plt.ylabel('Features',fontsize=18)
    plt.show()
    return 
   
def grid_search(X_train, y_train):
    """
    Grid search the parameters such as the max depth of the tree, the over sampling ratio, class weight.
    
    INPUT:
    X_train, DataFrame,  training features
    Y_train, array, training labels
    
    OUTPUT:
    clf:  the best model after grid search
    
    """
    
    k_values = [3, 4, 5, 6]
    up_ratio = np.linspace(0.3,0.6,4)
    weights = [1, 5, 10, 20]
    max_depth = [3,4,5,6,7]
    parameters = {'over__sampling_strategy':up_ratio,
                  'over__k_neighbors': k_values,
                  'model__max_depth': max_depth,
                  'model__scale_pos_weight':weights}
    model = XGBClassifier(random_state=42)
    over = SMOTE(random_state=42)
    steps = [('over', over), ('model', model)]
    pipeline = Pipeline(steps=steps)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    clf = GridSearchCV(pipeline, parameters, scoring='roc_auc', cv=cv)
    clf.fit(X_train, y_train)
    print(clf.best_score_) 
    print(clf.best_params_)
    return clf

def evaluate_test_dataset(X_test, y_test, clf):
    """
    Evaluate the model performance by printing the PR AUC, recall, precision, f1 score
    and plot the confusion matrix
    
    INPUT:
    X_test, DataFrame,  training features
    Y_test, array, training labels
    
    OUTPUT:
    None
    """
    y_prob = clf.predict_proba(X_test)
    pos_probs = y_prob[:, 1]
    precision, recall, thres= precision_recall_curve(y_test, pos_probs)
    print('PR AUC:',auc(recall, precision))
    print('')
    y_pred = pos_probs * 0
    y_pred[pos_probs> 0.2] = 1
    target_names = ['class 0', 'class 1']
    print(classification_report(y_test, y_pred, target_names=target_names))
    # plot confusion matrix
    df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index = ['0', '1'],
                  columns = ['0', '1'])
    plt.figure(figsize = (5,3))
    sns.heatmap(df_cm, annot=True, fmt="d")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

