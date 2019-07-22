import pandas as pd
import numpy as np

def print_if_verbose(statement, verbose=True):
    if verbose:
        print(statement)
    return


def columns2drop(df):
    "This function returns the names of the columns containing null values"
    rm = df.isnull().sum(axis=0)
    return rm[rm != 0].index.values


def detect_variables_type(clean_df):
    print('automatically detecting categorical variables:')
    numerical_vars = clean_df.select_dtypes(exclude=object).columns.values.tolist()
    categorical_vars = clean_df.select_dtypes(include=object).columns.values.tolist()
    print('numerical variables:\n%s' % str(numerical_vars))
    print('categorical variables:\n%s\n' % str(clean_df.select_dtypes(include=object).columns.values))

    return numerical_vars, categorical_vars


def prepare_dataset(dataset_path, labels, columns_type_dataset=None, verbose=True):
    """This function automatically prepare the dataset for the future computations

    WARNING: the file extension must be either csv or json

    * dataset_path: str, name of the dataset (if in the same folder) or dataset path

    * labels: int or list, The number of labels associated to each instance,if int the function will look a the last "labels" columns of the dataframe,
            if type list the function will select the names in the list as the columns containing the labels

    * columns_type_dataset: pyhton dict, it has two keys "categorical_vars" and "numerical_vars" and each corrisponding value is a list of the names
                            of the columns which contains categorical variables and numerical variables
                            e.g. {'categorical_vars':['categ_1','categ_2', ... ,'categ_n'],'numerical_vars':['num_1','num_2', ... ,'num_n']}
    """
    columns_type_dataset = columns_type_dataset or {}

    # reading the file
    if '/' in dataset_path:
        dataset_name = dataset_path.split('/')[-1]
    else:
        dataset_name = dataset_path
    print_if_verbose('reading dataset %s' % dataset_name, verbose)

    file_extension = dataset_name.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(dataset_path, sep=',')
    elif file_extension == 'json':
        df = pd.read_json(dataset_path)

    # checking if the labels are correct and if they contains null values
    if type(labels) == int:
        Y = df.iloc[:, -labels:].copy()
    elif type(labels) == list:
        Y = df[labels].copy()
    else:
        raise ValueError('"labels" must be an int or a string')
        return

    labels_name = Y.columns.values
    print_if_verbose('labels names: \n%s' % str(labels_name), verbose)
    labels_to_drop = columns2drop(Y)
    clean_Y = Y.drop(labels_to_drop, 1)
    print_if_verbose(f'dropping {len(labels_to_drop):d} columns:\n{str(labels_to_drop)}\nleaving {len(clean_Y.columns):d} columns:\n{str(clean_Y.columns.values)}\n', verbose)

    # cleaning from null values
    print_if_verbose('drop all columns containing null values (we prefer to keep instances rather than features)',
                     verbose)
    df.drop(labels_name, 1, inplace=True)
    columns2drop_names = columns2drop(df)
    clean_df = df.drop(columns2drop_names, axis=1).reset_index().drop('index', 1).copy()
    print_if_verbose(f'dropping {len(columns2drop_names):d} columns:\n{str(columns2drop_names)}\nleaving {len(df.columns) - len(columns2drop_names):d} columns:\n{str(clean_df.columns.values)}\n', verbose)

    # check if the dictionary contaning the name of the categorical and numerical variables is not empty
    if columns_type_dataset:
        # if this condition is true the dictionary is not empty
        print_if_verbose('the dict is not empty', verbose)
        if len(columns_type_dataset['categorical_vars']) > 0:
            # if there are categorical variables:
            print_if_verbose('%d categorical variables from the dict: \n%s' % (len(columns_type_dataset['categorical_vars']), str(columns_type_dataset['categorical_vars'])), verbose)

            categorical_vars = list(set(columns_type_dataset['categorical_vars']) & set(clean_df.columns.tolist()))
            numerical_vars = list(set(columns_type_dataset['numerical_vars']) & set(clean_df.columns.tolist()))
            print_if_verbose('%d categorical varibales after dropping null values:\n%s' % (len(categorical_vars), str(categorical_vars)), verbose)

            clean_df[categorical_vars] = clean_df[categorical_vars].astype('category')
            df_encoded = pd.get_dummies(clean_df, categorical_vars, prefix_sep='=')
            new_categorical_vars = df_encoded.drop(numerical_vars, 1).columns.values.tolist()
        else:
            print_if_verbose('there are no categorical variables', verbose)
            df_encoded = clean_df.copy()
            new_categorical_vars = []
            numerical_vars = list(set(columns_type_dataset['numerical_vars']) & set(clean_df.columns.values.tolist()))
    else:
        print_if_verbose('the dict is empty', verbose)
        # it the condition is false than the dictionary is empty
        #features:
        numerical_vars, categorical_vars = detect_variables_type(clean_df)
        df_encoded = pd.get_dummies(clean_df, categorical_vars, prefix_sep='=')
        new_categorical_vars = df_encoded.drop(numerical_vars, 1).columns.values.tolist()

        # labels:
        numerical_labels, categorical_labels = detect_variables_type(clean_Y)
        Y_df_encoded = pd.get_dummies(clean_Y, categorical_labels, prefix_sep='=')
        labels_name = Y_df_encoded.drop(numerical_labels, 1).columns.values.tolist()

    final_df = pd.concat([df_encoded,Y_df_encoded],axis=1)
    return final_df, numerical_vars, labels_name, new_categorical_vars
