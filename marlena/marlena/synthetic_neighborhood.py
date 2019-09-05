import pandas as pd
import numpy as np
from sklearn.metrics import hamming_loss, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
#from imblearn.over_sampling import RandomOverSampler
#from .rules import istance_rule_extractor


def sample_alphaFeatLabel(samplekNN_feat, samplekNN_label, alpha, random_state):
    
    """
    This function takes two sample of X2E instances taken according to:
            * their distance from the instance to be explained (i2e) in the FEATURE SPACE; samplekNN_feat
            * their distance from the instance to be explained (i2e) in the LABEL SPACE; samplekNN_label
            
    :param samplekNN_feat: pandas.DataFrame, it contains the features from the k nearest neighbors of the instance to be explained in the feature space 
    :param samplekNN_label: pandas.DataFrame, it contains the features from the k nearest neighbors of the instance to be explained in the label space
    :param alpha: double, default 0.7, % of samples to be taken from samplekNN_feat and (1-alpha from the label space)
    :param random_state: the random seed to be used when sampling
    :return: pandas.DataFrame, containing the k nearest neighbors of the instance to be explained according to mixed MARLENA procedure
    """
    #print('sample_alphaFeatLabel')
    #print(f'mixed selected with alpha = {alpha}, k = {len(samplekNN_feat)}')
    subsample_knn_feat = samplekNN_feat.sample(frac=alpha,random_state=random_state).reset_index().drop('index',1)
    #print(f'len(subsample_knn_feat) = {len(subsample_knn_feat)}')
    subsample_knn_label = samplekNN_label.sample(frac=(1-alpha),random_state=random_state).reset_index().drop('index',1)
    #print(f'len(subsample_knn_label) = {len(subsample_knn_label)}')
    alpha_sample_knn = pd.concat([subsample_knn_feat,subsample_knn_label]).reset_index().drop('index',1)
    #print(f'len(alpha_sample_knn) = {len(alpha_sample_knn)}')

    if len(alpha_sample_knn)<len(samplekNN_feat):
        n = len(samplekNN_feat)-len(alpha_sample_knn)
        if alpha > (1-alpha):
            alpha_sample_knn = pd.concat([alpha_sample_knn,samplekNN_feat.sample(n=n, random_state=random_state).reset_index().drop('index',1)])
        else:
            alpha_sample_knn = pd.concat([alpha_sample_knn,samplekNN_label.sample(n=n, random_state=random_state).reset_index().drop('index',1)])

        alpha_sample_knn = alpha_sample_knn.reset_index().drop('index',1)
    #print(f'alpha_sample_knn: {alpha_sample_knn.head(10)}, {alpha_sample_knn.tail(10)}')
    return alpha_sample_knn


def random_synthetic_neighborhood_df(sample_Knn, size, categorical_var, numerical_var, bb, labels_name, random_state):
    """
    :param sample_Knn: pandas.DataFrame, core of real neighbors selected using the mixed or the union approach
    :param size: int, the number of synthetic instances to generate
    :param categorical_var: list, name of columns containing discrete variables
    :param numerical_var: list, name of columns containing continuos variables
    :param labels_name: list, name of columns containing the classes labels
    :param random_state: the random seed to be used when sampling
    :return: pandas.Dataframe, the synthetic neighborhood + real core neighborhood
    """
    #print('random_synthetic_neighborhood_df!')
    df = sample_Knn#.drop(labels_name,1)
    #print(f'sample_Knn.head(): {sample_Knn.head()}')
    #np.random.seed(random_state)

    #print(f'len(continuous_var): {len(numerical_var)}')
    if len(numerical_var)>0:
        #print('there are continuos variables')
        cont_cols_synthetic_instances = list()
        for col in numerical_var:
            values = df[col].values
            mu = np.mean(values)
            sigma = np.std(values)
            new_values = random_state.normal(mu, sigma, size)
            #new_values = np.random.normal(mu,sigma, size)
            cont_cols_synthetic_instances.append(new_values)

        cont_col_syn_df = pd.DataFrame(data=np.column_stack(cont_cols_synthetic_instances), columns=numerical_var)

    #print(f'len(discrete_var): {len(categorical_var)}')
    if len(categorical_var)>0:
        #print('there are discrete variables')
        disc_cols_synthetic_instances = list()
        for col in categorical_var:
            values = df[col].values.tolist()
            diff_values = np.unique(values)
            prob_values = [values.count(val) / len(values) for val in diff_values]
            new_values = random_state.choice(diff_values, size, prob_values)
            #new_values = np.random.choice(diff_values, size, prob_values)
            disc_cols_synthetic_instances.append(new_values)

        disc_col_syn_df = pd.DataFrame(data=np.column_stack(disc_cols_synthetic_instances), columns=categorical_var)

    if (len(numerical_var) > 0)&(len(categorical_var) > 0):
        #I add here the original knn
        #print('both continuous and discrete varibles')
        synth_neighs = pd.concat([cont_col_syn_df,disc_col_syn_df],axis=1)
        synth_neighs_plus_knn = pd.concat([synth_neighs, sample_Knn.loc[:, numerical_var + categorical_var]])
        #print(f'sample_Knn.index: {sample_Knn.index}')
        return synth_neighs_plus_knn

    elif len(numerical_var)==0:
        #print('there are no continuous variables')
        #print(f'disc_col_syn_df: {disc_col_syn_df.head()}\n')
        final_sample = sample_Knn.drop('distance', 1).drop(labels_name, 1)
        #print(f"sample_Knn: {final_sample.head()} \n")

        return pd.concat([disc_col_syn_df, final_sample])#sample_Knn.loc[:, numerical_var + categorical_var]])

    elif len(categorical_var)==0:
        #print('there are no discrete variables')
        final_sample = sample_Knn.drop('distance', 1).drop(labels_name, 1)
        return pd.concat([cont_col_syn_df, final_sample])
    else:
        print('Error, no variables in input df')



def tuned_tree(self, X,y,param_distributions,scoring='f1_micro',cv=5):
    """
    This function performs an hyperpatameter tuning using a randomized search (see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
    and returns a tuned decision tree
    
    :param X: array-like or sparse matrix, shape = [n_samples, n_features], the training input sample, synthetic neighborhood features
    :param y: array-like, shape = [n_samples] or [n_samples, n_outputs], the target values (class labels) as integers or strings, the labels of the black box
    :param param_distributions:  dict, dictionary with parameters names (string) as keys and distributions or lists of parameters to try. 
    :param scoring: string, callable, list/tuple, dict or None, default 'f1_micro'
    :param cv:  int (number of folds), cross-validation generator or an iterable, optional, it determines the cross-validation splitting strategy, default=5
    :return: sklearn DecisionTreeClassifier
    """

    def hamming_score(y_true, ypred):
        hamming = 1 - hamming_loss(y_true, ypred)
        return hamming

    hamming_scorer = make_scorer(hamming_score)

    tree = DecisionTreeClassifier(class_weight="balanced")
    sop = np.prod([len(v) for k, v in param_distributions.items()])
    n_iter_search = min(100, sop)
    random_search = RandomizedSearchCV(tree, param_distributions=param_distributions,scoring=hamming_scorer,n_iter=n_iter_search, cv=cv, random_state=self.random_state)#scoring='f1_micro', n_iter=n_iter_search, cv=cv)
    random_search.fit(X, y)
    best_params = random_search.best_params_
    tree.set_params(**best_params)
    tree.fit(X,y)
    
    return tree

#def oneDTforlabelpreds(i2e, synthetic_neigh, knn_neigh, cols_X, cols_Y_BB, param_distributions):
#    """
#    This function takes:
#    - i2e: pd.Series, the instance to be explained
#    - synthetic_neigh: pd.DataFrame, the synthetic neighborhood of i2e
#    - knn_neigh: pd.DataFrame, the core of real neighbors of i2e
#    - cols_X: list, names of the columns containing the features
#    - cols_Y_BB: list, names of the columns containing the labels
#    - param_distributions: dict, dictionary with parameters names (string) as keys and distributions or lists of parameters to try for the hyperparameter tuning of the DTs
#
#    and computes a DT that is trained to learn one of the labels, than it performs a concatenations of all the predictions
#    and returns:
#
#    - unionDT_labels_syn: pd.DataFrame
#    - unionDT_labels_knn: pd.DataFrame
#    - unionDT_labels_i2e: pd.Series
#    - DT_rules: list, rules, one for each label
#    - DT_rules_len: list, lenght of rules
#
#    """
#    DT_syn_labelspred = {}
#    DT_knn_labelspred = {}
#    DT_i2e_labelspred = {}
#    DT_rules = []
#    DT_rules_len = []
#    size = len(synthetic_neigh)
#
#    for label in synthetic_neigh[cols_Y_BB].columns.values:
#        label_col = synthetic_neigh[label]
#        not_dummy = len(label_col.drop_duplicates())>1
#        imbalanced = label_col.value_counts(normalize=True).iloc[0]>0.8
#        columns_DT = np.append(cols_X,label)
#        y_i2e_bb_label = i2e[label]
#
#        #if not all in one class (0 or 1)
#        if not_dummy:
#            X = synthetic_neigh[cols_X].values
#            y = synthetic_neigh[label].values
#
#            #if the % of instances belonging to the majority class is >80%
#            if imbalanced:
#                #oversampling with 'auto' (not majority) sampling strategy
#                ros = RandomOverSampler(sampling_strategy='auto',random_state=0)
#                X_resampled, y_resampled = ros.fit_resample(X, y)
#                #it creates a balanced (50/50) syn neigh with set size (1000)
#                set_size_syn_neigh = pd.DataFrame(np.c_[X_resampled, y_resampled],columns=columns_DT).groupby(label, group_keys=False).apply(lambda x: x.sample(int(size/2),random_state=0)).reset_index().drop('index',1)
#                X_resampled = set_size_syn_neigh[cols_X].values
#                y_resampled = set_size_syn_neigh[label].values
#
#                #fine tuning of the DT
#                tree = tuned_tree(X_resampled,y_resampled,param_distributions=param_distributions)
#                #training the DT
#                tree.fit(X_resampled, y_resampled)
#
#            else:
#                #not imbalanced
#                #fine tuning of the DT
#                tree = tuned_tree(X,y,param_distributions=param_distributions)
#                #training the DT
#                tree.fit(X, y)
#
#            ########################################
#            #DT prediction on the synthetic neighborhood (y_tree1_syn1)
#            DT_syn_labelspred[label]= tree.predict(X)
#            #DT prediction on the real neighborhood (y_tree1_kNN1)
#            y_samplekNN = knn_neigh[cols_Y_BB]
#
#            y_samplekNN = knn_neigh[label].values
#            DT_knn_labelspred[label] = tree.predict(knn_neigh[cols_X].values)
#            ########################################
#
#            #rule tree1
#            rule_tree,len_rule = istance_rule_extractor(i2e[cols_X].values.reshape(1, -1),tree,cols_X)
#            rule_tree = rule_tree + ' ('+label+')'
#            DT_rules.append(rule_tree)
#            DT_rules_len.append(len_rule)
#            #prediction of i2e
#            DT_i2e_labelspred[label] = tree.predict(i2e[cols_X].values.reshape(1, -1))
#
#        else: #if the column is dummy the tree will be dummy, I'm not growing a tree, it's a waste of time
#            #print('%s is dummy:' %label)
#            X = synthetic_neigh[cols_X].values
#            y = synthetic_neigh[label].values
#            rule_tree = '->['+str(int(y[0]))+']'+' ('+label+')'
#            len_rule = 0
#
#            DT_rules.append(rule_tree)
#            DT_rules_len.append(len_rule)
#            DT_syn_labelspred[label]=y
#            DT_knn_labelspred[label]=y[0:len(knn_neigh)]
#            #prediction of i2e
#            DT_i2e_labelspred[label] = y[0]
#
#    unionDT_labels_syn = pd.DataFrame(DT_syn_labelspred)
#    unionDT_labels_knn = pd.DataFrame(DT_knn_labelspred)
#    unionDT_labels_i2e = pd.DataFrame(DT_i2e_labelspred,index=[0])
#
#    return unionDT_labels_syn, unionDT_labels_knn, unionDT_labels_i2e, DT_rules, DT_rules_len
