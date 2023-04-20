import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import math

import sklearn
from einsteinds import utils
from einsteinds import db
from einsteinds import event_processing

from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier

import xgboost
from xgboost import XGBClassifier


RANDOM_SEED = 1984
LABELS = ["Normal", "Fraud"]

def train_autoencoder(df, fraud_column):
    '''OLD: Needs to be updated.
    
    Arguments:
        df {pandas.DataFrame} -- The dataframe containing the records to train the autoencoder on.
        fraud_column {string} -- The string name of the column containin the fraud labels.
    
    Returns:
        keras.Model -- The keras neural network autoencoder model.
    '''

    drop_cols = ['fraud']
    data = df.copy()
    data['fraud'] = fraud_column
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
    X_train = X_train[X_train.fraud == 0]
    X_train = X_train.drop(drop_cols, axis=1)
    y_test = X_test['fraud']
    X_test = X_test.drop(drop_cols, axis=1)
    X_train = X_train.values
    X_test = X_test.values
    X_train.shape

    input_dim = X_train.shape[1]
    encoding_dim = int(input_dim/2)

    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="tanh", 
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
    decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    nb_epoch = 20
    batch_size = 32
    autoencoder.compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath="models/anomaly_autoencoder.h5",
                                verbose=0,
                                save_best_only=True)

    history = autoencoder.fit(X_train, X_train,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        verbose=1,
                        callbacks=[checkpointer]).history

    return autoencoder


def autoencoder_prediction(scaled_data, autoencoder, threshold=0.1):
    '''OLD: Makes a prediction based off a scaled dataset using an autoencoder model.
    
    Arguments:
        scaled_data {pandas.DataFrame} -- 
        autoencoder {keras.Model} -- The autoencoder model to use for prediction.
    
    Keyword Arguments:
        threshold {float} -- The threshold at which to classify an example as an anomaly (default: {0.1})
    
    Returns:
        pandas.DataFrame -- The results of the prediction using the autoencoder.
    '''

    pred = autoencoder.predict(scaled_data)
    squared_error = np.power(scaled_data - pred, 2)
    mse = np.mean(squared_error, axis=1)
    max_squared_error = np.apply_along_axis(arr=squared_error, axis=1, func1d=np.max)
    col_of_biggest_squared_error = np.array(scaled_data.columns[np.apply_along_axis(arr=squared_error, axis=1, func1d=np.argmax)])

    return pd.DataFrame({'anomaly_score_autoencoder': mse, 'biggest_anomaly_score_autoencoder': max_squared_error, 'biggest_anomaly_metric_autoencoder': col_of_biggest_squared_error})


def train_isolation_forest(data, contamination=0.10):
    '''Trains a scikit learn isolation forest anomaly detection model using the data.
    
    Arguments:
        data {pandas.DataFrame} -- The data to use for training the autoencoder.
    
    Keyword Arguments:
        contamination {float} -- The proportion of outliers in the dataset. (default: {0.10})
    
    Returns:
        sklearn.ensemble.IsolationForest -- The trained isolation forest model.
    '''

    ifst = sklearn.ensemble.IsolationForest()
    ifst.fit(data)
    
    return ifst

def isolation_forest_predictions(data , isolation_forest):
    '''Make predictions from the isolation forest model.
    
    Arguments:
        data {pandas.DataFrame} -- the dataset to predict anomoalies with
        isolation_forest {sklearn.ensemble.IsolationForest} -- the isolation forest model to use for generating predictions
    
    Returns:
        numpy.ndarray -- The array containing True, or False predcitions based on the data.
    '''


    anomaly = isolation_forest.predict(X=data) == -1

    return anomaly

def prepare_for_prediction(new_data, model_features):
    '''Prepares a new dataset for prediction with a model, by dropping any columns 
    that don't exist in the model's features and adding columns of zeros for features, the model
    requires, but that aren't in the new dataset.
    
    Arguments:
        new_data {pandas.DataFrame} -- The data to predict on.
        model_features {list[string]} -- The list of string column names of features in the trained model.
    
    Returns:
        pandas.DataFrame -- The new data reformatted to be used to generate predictions.
    '''

    new_data = new_data.copy()
    
    missing_features = [col for col in model_features if col not in new_data.columns]

    for col in missing_features:
        new_data[col] = 0

    new_data = new_data[model_features]
    
    return new_data.replace([np.nan, np.inf],0)


def split(group_counts, n_splits, tolerance=0):
    '''Given a target number of splits and a list of groups with a number
    of examples per group, greedily splits the data into n_splits such that all examples for a given group
    only exist in a single group and such that the number of examples in each group is as
    balanced as possible.
    
    Arguments:
        group_counts {pandas.DataFrame} -- The list of groups with count of examples per group.
        n_splits {int} -- The number of splits of the data.
    
    Keyword Arguments:
        tolerance {float} -- The allowable proportional difference in the number of examples per group.  (default: {0})
    
    Returns:
        dict -- A dictionary containing the split of groups.
    '''

    
    max_split_size = math.ceil(group_counts['count'].sum()/n_splits)
    
    splits = {}
    
    # initialize the dict
    for split in range(n_splits):
        splits[split] = {'groups': [], 'n_examples': 0}
    
    for i, (group, count) in enumerate(zip(group_counts['group'].values, group_counts['count'].values)):
        
        rand_splits = np.random.choice(np.arange(n_splits), size=n_splits, replace=False)
        
        for split in rand_splits:
            
            new_count = splits[split]['n_examples']+count
            
            # if the current split is empty and the count is less than tolerance over the max_split_size
            if count > max_split_size and splits[split]['n_examples'] == 0 and (count-max_split_size)/max_split_size <= tolerance:
            
                # if new_count > max_split_size and (new_count-max_split_size)/max_split_size <= tolerance:
                splits[split]['groups'].append(group)
                splits[split]['n_examples'] += count
                
                break
            else:
                if (max_split_size-splits[split]['n_examples']) >= count:
                    splits[split]['groups'].append(group)
                    splits[split]['n_examples'] += count
                    break
    
    
    return splits


def balanced_group_n_fold(labels, groups, tolerance=0.1, max_splits=10):
    '''Given a set of binary lables, and group for each label. Generates a splitting of the data
    for cross validation purposes that balances the number of examples in each split, but ensures all the records for a single
    group exist only in the training or test split and not both, to avoid issues of data contamination.

    Note that an important feature here is that the number of splits is determined by the data, up to max_splits. 
    So if there is only two possible splits that balance the data, only two splits will be returned.
    
    Arguments:
        labels {numpy.array} -- The true/false labels of the data.
        groups {numpy.array} -- The array containing the groups for each example label.
    
    Keyword Arguments:
        tolerance {float} -- The allowable proportional difference in the number of examples per group. (default: {0.1})
        max_splits {int} -- The maximum number of splits to create. (default: {10})
    
    Returns:
        list[tuple(numpy.array, numpy.array)] -- A list of tuples containing the indices of the training and test set for each split.
    '''

    
    # determine the minority group class
    df = pd.DataFrame({'label': labels, 'group': groups})
    
    # count the number of groups by class
    result = df.groupby('label')['group'].agg(['count']).sort_values(by='count', ascending=True).reset_index()
    
    # select the class with the fewest groups
    minority_group_class = result['label'].values[0]
    majority_group_class = result['label'].values[1]
    
    minority_df = df[df['label'] == minority_group_class]
    majority_df = df[df['label'] == majority_group_class] # should check that there is indeed a second class

    #print("Minority Class: ", minority_group_class)
    
    # determine the max number of folds
    # get the group with the most number of examples in the minority class
    minority_group_counts = minority_df.groupby('group', as_index=False)['label'].agg(['count']).reset_index().sort_values(by='count', ascending=False)
    majority_group_counts = majority_df.groupby('group', as_index=False)['label'].agg(['count']).reset_index().sort_values(by='count', ascending=False)

    max_folds = minority_df.shape[0]/minority_group_counts['count'].values[0]
    max_folds = math.ceil(max_folds) if max_folds%1 >= (1-tolerance) else math.floor(max_folds)
    max_folds = min(max_folds, max_splits)
    #print("Number of folds is: ",max_folds)
    
    minority_splits = split(minority_group_counts, max_folds, tolerance=tolerance)
    majority_splits = split(majority_group_counts, max_folds, tolerance=tolerance)
    
    combined = {}
    
    # combine the minority and majority groups together
    for key in minority_splits.keys():
        combined[key] = {'groups': minority_splits[key]['groups'] + majority_splits[key]['groups'],
                         'n_examples': minority_splits[key]['n_examples'] + majority_splits[key]['n_examples']}
        
    # for key in combined.keys():
    #     print("Minority Class Fold:", key, ', groups:', len(minority_splits[key]['groups']), ', n_examples:', minority_splits[key]['n_examples'])
    #     print("Majority Class Fold:", key, ', groups:', len(majority_splits[key]['groups']), ', n_examples:', majority_splits[key]['n_examples'])
    #     print("Combine Fold:", key, ', groups:', len(combined[key]['groups']), ', n_examples:', combined[key]['n_examples'])
    
    splits = []
    
    # set up the indices
    indices = np.arange(len(labels))
    
    # create the plit indices
    for key in combined.keys():
        splits.append((indices[np.isin(groups, combined[key]['groups']) == False],indices[np.isin(groups, combined[key]['groups'])]))
    
    result = splits
    
    return result


def crossValidationResults(df, labels, groups, classifier_list):
    '''Generates training metric results for a given dataset that contains binary labels, a grouping variable and a list of classifiers.
    The output contains a list of metrics for each clasiifier accross an n_fold cross validation.
    
    Arguments:
        df {pandas.DataFrame} -- The dataset
        labels {string} -- The string name of the column containing the labels.
        groups {string} -- The string name of the column containing the group labels.
        classifier_list {list} -- The list of classifiers to perform a cross validated training routine on.
    
    Returns:
        pandas.DataFrame -- The n_fold cross validation results.
    '''

    # variables to keep track of
    training_fraud_percentages = []
    testing_fraud_percentages = []
    training_accuracies = []
    testing_accuracies = []
    training_average_precisions = []
    testing_average_precisions = []
    train_true_negatives = []
    train_false_negatives = []
    train_true_positives = []
    train_false_positives = []
    test_true_negatives = []
    test_false_negatives = []
    test_true_positives = []
    test_false_positives = []
    classifiers = []
    
    X = df.drop([labels,groups], axis=1).astype('float32')
    X[X == -np.inf] = 0
    X[X == np.inf] = 0
    X = X.values
    
    y = df[labels].values
    groups = df[groups].values

    cv_folds = balanced_group_n_fold(y, groups)
    
    # for each classifer type
    for clf in classifier_list:
        
        classifier = clf['Label']
        model = clf['Model']

        for train_index, test_index in cv_folds:

            X_train = X[train_index,:]
            X_test = X[test_index,:]
            y_train = y[train_index]
            y_test = y[test_index]

            class_ratio = np.sum(y_train == 0)/np.sum(y_train == 1)

            if type(model) == type(XGBClassifier()):
                model.set_params(**{'scale_pos_weight': class_ratio})

            model.fit(X_train, y_train)

            preds_train = model.predict(X_train)
            preds_test = model.predict(X_test)
            probs_train = model.predict_proba(X_train)[:,1]
            probs_test = model.predict_proba(X_test)[:,1]

            training_accuracy = np.sum(preds_train == y_train)/len(y_train)
            testing_accuracy = np.sum(preds_test == y_test)/len(y_test)

            training_avp = sklearn.metrics.average_precision_score(y_train, probs_train)
            testing_avp = sklearn.metrics.average_precision_score(y_test, probs_test)

            train_cm = sklearn.metrics.confusion_matrix(y_train, preds_train, sample_weight=None)

            test_cm = sklearn.metrics.confusion_matrix(y_test, preds_test, sample_weight=None)

            training_fraud_percentages.append(np.sum(y_train)/len(y_train))
            testing_fraud_percentages.append(np.sum(y_test)/len(y_test))
            training_accuracies.append(training_accuracy)
            testing_accuracies.append(testing_accuracy)
            training_average_precisions.append(training_avp)
            testing_average_precisions.append(testing_avp)
            train_true_negatives.append(train_cm[0,0])
            train_false_negatives.append(train_cm[1,0])
            train_true_positives.append(train_cm[1,1])
            train_false_positives.append(train_cm[0,1])
            test_true_negatives.append(test_cm[0,0])
            test_false_negatives.append(test_cm[1,0])
            test_true_positives.append(test_cm[1,1])
            test_false_positives.append(test_cm[0,1])
            classifiers.append(classifier)

    
    # put results together in a dataframe
    model_results_df = pd.DataFrame({
        'classifier': classifiers,
        'train_fraud_percentage': training_fraud_percentages,
        'test_fraud_percentage': testing_fraud_percentages,
        'train_accuracy': training_accuracies,
        'test_accuracy': testing_accuracies,
        'train_average_precision': training_average_precisions,
        'test_average_precision': testing_average_precisions,
        'train_true_negatives': train_true_negatives,
        'train_false_negatives': train_false_negatives,
        'train_true_positives': train_true_positives,
        'train_false_positives': train_false_positives,
        'test_true_negatives': test_true_negatives,
        'test_false_negatives': test_false_negatives,
        'test_true_positives': test_true_positives,
        'test_false_positives': test_false_positives,
    })     

    # calculate precision and recall
    model_results_df['test_precision'] = model_results_df.test_true_positives/(model_results_df.test_true_positives+model_results_df.test_false_positives)
    model_results_df['test_recall'] = model_results_df.test_true_positives/(model_results_df.test_true_positives+model_results_df.test_false_negatives)
    model_results_df['train_precision'] = model_results_df.train_true_positives/(model_results_df.train_true_positives+model_results_df.train_false_positives)
    model_results_df['train_recall'] = model_results_df.train_true_positives/(model_results_df.train_true_positives+model_results_df.train_false_negatives)
    model_results_df.fillna(0, inplace=True)

    return model_results_df[sorted(model_results_df.columns)]


def cross_val_avp(classifier, df):
    '''Given  a single classifier and a dataset, return the mean of the testing average precision scores over an n_fold cross validation.
    
    Arguments:
        classifier {sklearn model} -- The classifier to train.
        df {pandas.DataFrame} -- The dataframe to use for training.
    
    Returns:
        float -- The mean testing average precision score from the n_fold cross validation.
    '''


    print("Computing mean average precision score")
    model_results = crossValidationResults(df=df, labels='fraud', groups='user_email', classifier_list=[{'Label': 'rf', 'Model': classifier}])
    summary = model_results.groupby(['classifier'], as_index=False)[['test_precision','test_recall','test_average_precision']].agg(['mean','median','std']).reset_index()
    summary.columns = [col[0] if col[1] == '' else '_'.join(col) for col in summary.columns.ravel()]
    
    return summary.to_dict('records')[0]['test_average_precision_mean']


def generate_optimal_random_forest(requests_summary_with_fraud):
    '''Uses scikit optimize to build a model with optimized hyperparameters 
    by using bayesian hyperparamater optimization and n_fold cross validation.
    
    Arguments:
        requests_summary_with_fraud {pandas.DataFrame} -- The dataframe containing the training data with a fraud column.
    
    Returns:
        dict(classifier=sklearn.ensemble.RandomForest, model_features=list[string]) -- A dict containing the trained optimized model and the list of features used to create it.
    '''

    print("Generating an optimized random forest, using bayesian optimization.")
    # make sure there's no na or inf values
    requests_summary_with_fraud = requests_summary_with_fraud.replace([np.nan, np.inf, -np.inf], 0)

    # initialize the base classifier
    clf = RandomForestClassifier()
    
    # get the max number of features
    n_features = requests_summary_with_fraud.shape[1] - 2

    # define the search space for the classifier
    space  = [Integer(1, 10, name='max_depth'),
              Integer(10, 100, name='n_estimators'),
              Integer(10, n_features, name='max_features')]

    # define the objective function to minimize
    @use_named_args(space)
    def objective(**params):
        clf.set_params(**params)

        return -np.mean(cross_val_avp(clf, requests_summary_with_fraud))
    
    # use bayesian hyperparameter optimization to tune the best model
    res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)
    
    print("Model Average Precision Score", res_gp.fun)
    
    print("""Best parameters:
    - max_depth=%d
    - n_estimators=%d
    - max_features=%d
    """ % (res_gp.x[0], res_gp.x[1], res_gp.x[2]))
    
    # get the best parameters
    max_depth = res_gp.x[0]
    n_estimators = res_gp.x[1]
    max_features = res_gp.x[2]
    
    # train on the full dataset
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, max_features=max_features)

    # extract the features for later data preparation
    features = requests_summary_with_fraud.drop(['user_email','fraud'], axis=1).columns
    X = requests_summary_with_fraud.drop(['user_email','fraud'], axis=1).values
    y = requests_summary_with_fraud.fraud.values
    
    # fit the model with the full dataset
    rf.fit(X=X, y=y)
    
    preds = rf.predict(X)
    
    print("Average Precision Score on full training set", average_precision_score(y_true=y, y_score=preds))
    
    return {'classifier': rf, 'model_features': features}


def save_random_forest_model(model, path):
    '''Saves a random forest model to file as a pickle.
    
    Arguments:
        model {dict} -- The model definition and feature list.
        path {string} -- The location to save the model.
    '''

    
    pickle.dump(model, open(path, "wb"))


def load_random_forest_model(path):
    '''Loads a random forest model from a given pickel file.
    
    Arguments:
        path {string} -- the path of the pickle file.
    
    Returns:
        dict -- The model definition and feature list.
    '''

    model = pickle.load(open(path, "rb"))
    
    return model


def train_random_forest(start_date, end_date, db):
    '''Given a start and end date defining the time period to use as training data
    and a database connection, trains an optimal random forest model to predict credit card and interac
    fraud and save the model as a pickle file at location of the script calling this function.

    Arguments:
        start_date {datetime.datetime} -- The start date to bound the training data.
        end_date {datetime.datetime -- The end date to bound the training data.
        db {einsteinds.db.Database} -- The custom database object that wraps pymongo.
    '''


    # get all the request summaries in the date range
    request_summaries = db.get_summarized_request_sets(start_date=start_date, end_date=end_date)
    
    # get add the fraud labels from the current blacklist pulled from the database
    request_summaries_with_fraud = db.add_fraud_label(request_summaries, 'user_email')
    
    # generate the optimal random forest model
    model = generate_optimal_random_forest(request_summaries_with_fraud)
    
    # save the model as a pickle with the feature list
    save_random_forest_model(model, 'random_forest.p')


def predict_from_request(request, db, model, threshold=0.5):
    '''Given a single raw request, a database connection, a model definition and a threshold for classification,
    predicts if a request is fraudulent or not.
    
    Arguments:
        request {json} -- The raw event containing the request information.
        db {einsteinds.db.Database} -- The custom database object that wraps pymongo.
        model {dict} -- The model definition.
    
    Keyword Arguments:
        threshold {float} -- The threshold for classification. If the predicted probability is > threshold the example is True/fraudulent. (default: {0.5})
    
    Returns:
        bool -- True or False representing if the request is fraudulent or not.
    '''

    # generate the request set - this goes to the database and pulls the events related to the request
    # would be better to speed this up using a local copy of events
    request_set = db.get_deposit_request_set(request=request)
    
    # summarize the request sets
    request_set_summaries = db.get_summarized_request_sets(rsets=[request_set])
    
    # pull the model and features from the definition
    classifier = model['classifier']
    features = model['model_features']
    
    # prepare the request for prediction
    X = prepare_for_prediction(request_set_summaries, features)
    
    # generate the prediction
    predictions = classifier.predict_proba(X)[:,1]
    
    # return the fraud result
    return predictions[0] > threshold


def predict_from_clean_request_sets(request_sets, db, model, threshold=0.5):
    '''Given a list of cleaned request sets, a database connection, a model definition and a threshold for classification,
    predicts if each request is fraudulent or not.
    
    Arguments:
        request_sets {list[json]} -- The clearned request sets to use for prediction.
        db {einsteinds.db.Database} -- The custom database object that wraps pymongo.
        model {dict} -- The model definition.
    
    Keyword Arguments:
        threshold {float} -- The threshold for classification. If the predicted probability is > threshold the example is True/fraudulent. (default: {0.5})
    
    Returns:
        pandad.DataFrame -- The prediction results.
    '''
    
    # summarize the request sets
    request_set_summaries = db.get_summarized_clean_request_sets(rsets=request_sets)
    
    # pull the model and features from the definition
    classifier = model['classifier']
    features = model['model_features']
    
    # prepare the request for prediction
    X = prepare_for_prediction(request_set_summaries, features)
    
    # generate the prediction
    predictions = classifier.predict_proba(X)[:,1]

    binary_predicion = predictions[0] > threshold

    results = request_set_summaries.reset_index()
    results['prediction_probability'] = predictions
    results['prediction_binary'] = binary_predicion
    
    # return the fraud result
    return results

