import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import f1_score
from sklearn.utils import check_random_state
from sklearn.tree import DecisionTreeClassifier

from .distances import mixed_distance_arrays, normalized_euclidean_distance, simple_match_distance
from .synthetic_neighborhood import sample_alphaFeatLabel, random_synthetic_neighborhood_df, tuned_tree
from .rules import istance_rule_extractor

class MARLENA(object):

    def __init__(self,
                 neigh_type='unified',
                 cat_dist=simple_match_distance,
                 num_dist=normalized_euclidean_distance,
                 random_state = None):
        """
        :param neigh_type: str, 'unified' or 'mixed'
        :param cat_dist: func(x,y), distance function to be used when evaluating distance between the categorical variables of x and y (two 1-D arrays), default: scipy hamming distance
        :param num_dist: func(x,y), distance function to be used when evaluating distance between the numerical variables of x and y (two 1-D arrays), default: normalized_euclidean_distance
        :param random_state: an integer or numpy.RandomState that will be used to generate random numbers. If None, the random state will be initialized using the internal numpy seed.
        """
        if neigh_type not in ['unified','mixed']:
            raise ValueError('neigh_type can be either "unified" or "mixed"')
        else:
            self.neigh_type = neigh_type
        self.cat_dist = cat_dist
        self.num_dist = num_dist
        self.random_state = check_random_state(random_state)


    def _label_X2E(self, i2e=None):
        """
        This function uses the black box to labels the instance(s)

        :param i2e: pandas.Series, default None,
                    it is the instance whose black box decision we want to explain
                    it should not already contain the labels, but if it does it is internally taken care of.
                    if None, implicitly takes a pandas.Dataframe, the dataset whose instances we want to use to generate the synthetic neighbors,
                    again, it should not already contain the labels, but if it does it is internally taken care of.
        :return: X2E: pandas.Dataframe or pandas.Series labelled by the black-box
        """
        if i2e is None:
            #if this is the dataframe of all instances
            #first thing we reset the index
            self.dataset_ = self.dataset_.reset_index(drop=True)
            #then we take only the features
            dataset2blabeled_df = self.dataset_.drop(self.labels_name_, 1, errors='ignore')
        else:
            #if this is a single instance
            #we take only the features
            dataset2blabeled_df = pd.DataFrame(i2e.drop(self.labels_name_, errors='ignore')).T

        #we then use the black box to label the instance(s)
        bb_labels_df = pd.DataFrame(self.blackbox_.predict(dataset2blabeled_df), columns=self.labels_name_)

        X2E = pd.concat([dataset2blabeled_df, bb_labels_df],axis=1)
        return X2E

    def _sample_knn(self, k, X2E, i2e, num_dist, cat_dist):

        """
        :param k: int, (core real neighborhood size) number of first neighbors to consider when sampling to generate the synthetic neighborhood
        :param X2E: pandas.DataFrame, (X2E_wlables) the dataset whose instances we want to use to generate the synthetic neighbors labelled by '_label_X2E'
        :param i2e: pandas.Series, (i2e_wlabels) it is the instance whose black box decision we want to explain labelled by '_label_X2E'
        :param num_dist: func(x,y), distance function to be used when evaluating distance between the numerical variables of x and y (two 1-D arrays), default: normalized_euclidean_distance
        :param cat_dist: func(x,y), distance function to be used when evaluating distance between the categorical variables of x and y (two 1-D arrays), default: simple match distance
        :return:
        """
        #print(f'_sample_knn with k= {k}, len(X2E) = {len(X2E)}')

        len_cat = len(self.categorical_vars_)
        #print(f'len(cat_vars) = {len_cat}')
        len_num = len(self.numerical_vars_)
        #print(f'len(num_vars) = {len_num}')
        
        if self.neigh_type == 'unified':
            #the distance is evaluated using one distance that takes into consideration both feature space and label space at the same time
            #print('unified')
            categorical_names = list(self.categorical_vars_) + list(self.labels_name_)
            i2e = i2e.loc[categorical_names + list(self.numerical_vars_)]
            #print(f'i2e: {i2e}')

            mydist = lambda x, y: mixed_distance_arrays(x,
                                                 y,
                                                 len_categorical=len(categorical_names),
                                                 len_numerical=len_num,
                                                 num_dist=num_dist,
                                                 cat_dist=cat_dist)
            # evaluate the distances between the instance to be explained and the other instances in the dataframe
            distances_i2e = pd.DataFrame(distance.cdist([i2e.tolist()], X2E, mydist).T, columns=['distance'])
            #print(f'distances_i2e.head()={distances_i2e.head()}')
            # take the first k instances
            kNN_unif_df = pd.concat([X2E, distances_i2e], 1).sort_values(by='distance', ascending=True).iloc[:k]
            #print(f'len(kNN_unif_df) = {len(kNN_unif_df)}')
            #print(f'head(kNN_unif_df):\n{kNN_unif_df.head()}')
            return kNN_unif_df
        
        elif self.neigh_type=='mixed': # if neigh_type == 'mixed'
            #print('mixed')
            #the distance is evaluated taking alpha neighbors from the feature space and 1-alpha neighbors from the label space
            i2e = i2e.loc[list(self.categorical_vars_) + list(self.numerical_vars_) + list(self.labels_name_)]
            #print(f'i2e: {i2e}')
            mydist = lambda x, y: mixed_distance_arrays(x,
                                                 y,
                                                 len_categorical=len_cat,
                                                 len_numerical=len_num,
                                                 num_dist=num_dist,
                                                 cat_dist=cat_dist)
            #paiwise distance in the label space
            labelspace_distances_i2e = pd.DataFrame(distance.cdist([i2e[self.labels_name_]], X2E[self.labels_name_], mydist).T, columns=['distance'])
            #print(f'labelspace_distances_i2e: {labelspace_distances_i2e.head()}')
            samplekNN_label = pd.concat([X2E, labelspace_distances_i2e], 1).sort_values(by='distance', ascending=True).head(k)
            #print(f'len(samplekNN_label): {len(samplekNN_label)}')
            #print(f'head(samplekNN_label):\n{samplekNN_label.head()}')
            #pairwise distance in the feature space
            #print(f'[i2e.drop(self.labels_name_)] = {[i2e.drop(self.labels_name_)]}')
            feature_space_distance_i2e = pd.DataFrame(distance.cdist([i2e.drop(self.labels_name_)], X2E.drop(self.labels_name_,1), mydist).T, columns=['distance'])
            #print(f'feature_space_distance_i2e: {feature_space_distance_i2e}')
            samplekNN_feat = pd.concat([X2E, feature_space_distance_i2e], 1).sort_values(by='distance', ascending=True).head(k)
            #print(f'len(samplekNN_feat): {len(samplekNN_feat)}')
            #print(f'head(samplekNN_feat):\n{samplekNN_feat.head()}')
            kNN_mixed_df = sample_alphaFeatLabel(samplekNN_feat, samplekNN_label, alpha=self.alpha, random_state=self.random_state)
            #print(f'len(kNN_mixed_df) = {len(kNN_mixed_df)}')
            return kNN_mixed_df
        else:
            raise ValueError('neigh_type should be either "unified" or "mixed"')

    def _generate_synthetic_neighborhood(self, k, size=1000, alpha=0.7):
        """
        :param k: int, (core real neighborhood size) number of first neighbors to consider when sampling to generate the synthetic neighborhood
        :param size: int, (synthetic neighborhood size) default 1000, number of synthetic neighbors to generate
        :param alpha: float, default 0.7, when MARLENA neigh_type='mixed' this is the fraction of neighbors to sample from the feature space
        :return:
        """
        #alpha is the fraction of neighbors in the feature space when marlena is mixed
        if type(alpha)!=float:
            raise ValueError('alpha should be a float')
        else:
            self.alpha = alpha
        X2E_wlables = self._label_X2E()
        label_i2e = pd.Series(self.blackbox_.predict(self.i2e_.values.reshape(1,-1))[0],index=self.labels_name_)
        i2e_wlabels = pd.concat([self.i2e_,label_i2e])

        kNN_df = self._sample_knn(k, X2E_wlables, i2e_wlabels, self.num_dist, self.cat_dist)
        #print('cicciopuzzo', kNN_df.head(20))

        synthetic_neighbors = random_synthetic_neighborhood_df(sample_Knn = kNN_df,
                                                               size = size,
                                                               categorical_var= self.categorical_vars_,
                                                               numerical_var= self.numerical_vars_,
                                                               bb=self.blackbox_,
                                                               labels_name=self.labels_name_,
                                                               random_state = self.random_state)
        self.synthetic_neighbors_ = synthetic_neighbors
        
        #return synthetic_neighbors

    def extract_explanation(self, i2e, dataset, blackbox, numerical_vars, categorical_vars, labels_name, k, size=1000, alpha=0.7):
        """
        :param i2e: pandas.Series, it is the instance whose black box decision we want to explain (it must contain ONLY the features, NOT the labels)
        :param dataset: pandas.DataFrame, it is the dataframe containing the instances that will be used to generate the synthetic ones (it must contain ONLY the features, NOT the labels)
        :param blackbox: multi-label classifier with 'fit(X,y)' and 'predict(x)' methods (sklearn-style)
        :param numerical_vars: list of str, list containing the names of the dataset columns containing the numerical variables
        :param categorical_vars: list of str, list containing the names of the dataset columns containing the categorical variables
        :param labels_name: list of str, list containing the names of the dataset columns containing the labels (classes names)
        :param k: int, (core real neighborhood size) number of first neighbors to consider when sampling to generate the synthetic neighborhood
        :param size: int, (synthetic neighborhood size) default 1000, number of synthetic neighbors to generate
        :param alpha: float, default 0.7, when MARLENA neigh_type='mixed' this is the fraction of neighbors to sample from the feature space
        :return: rule (str), fidelity(float), hit(float), DT (sklearn decision tree)
        """

        self.blackbox_ = blackbox
        self.i2e_ = i2e
        self.dataset_ = dataset
        self.numerical_vars_ = numerical_vars
        self.categorical_vars_ = categorical_vars
        self.labels_name_ = labels_name

        ### generates the sythetic neighbors
        self._generate_synthetic_neighborhood(k, size, alpha)

        ### train a DT with tuned hyperparameters on the synthetic neighbors
        if (len(self.numerical_vars_)>0)&(len(self.categorical_vars_)>0):
            cols_X = self.numerical_vars_ + self.categorical_vars_
        elif len(self.numerical_vars_)>0:
            cols_X = self.numerical_vars_
        else:
            cols_X = self.categorical_vars_

        X = self.synthetic_neighbors_.loc[:, cols_X]
        #before building a DT we add the instance to be explained in the synthetic neighborhood
        #X = X.append(i2e,ignore_index=True)

        #print(f'X.shape = {X.shape}')
        y_bb = self.blackbox_.predict(X)
        #print(f'bb.predict synthetic:{y_bb}\n')

        #TO USE Tuned Decision trees (does not guarantee deterministic results)
        #dt_params = {
        #    'max_depth': [None, 10, 20, 30, 40, 50, 70, 80, 90, 100],
        #    'min_samples_split': [2 ** i for i in range(1, 10)],
        #    'min_samples_leaf': [2 ** i for i in range(1, 10)],
        #}
        #DT = tuned_tree(self, X, y_bb, dt_params, scoring='accuracy', cv=5)
        #print('Decision Tree (not tuned)')
        DT = DecisionTreeClassifier(class_weight="balanced", random_state = self.random_state)
        DT.fit(X, y_bb)

        ### extract the decision rule from the decision tree:
        rule, instance_imporant_feat,len_rule = istance_rule_extractor(i2e.values.reshape(1, -1),
                                                                       DT,
                                                                       cols_X,
                                                                       self.labels_name_,
                                                                       categorical_vars=self.categorical_vars_,
                                                                       numerical_vars=self.numerical_vars_)

        ## evuluating fidelity and hit of the rule
        y_true = y_bb
        y_pred = DT.predict(X)
        fidelity = f1_score(y_true=y_true, y_pred=y_pred, average='micro')

        y_true = self.blackbox_.predict(i2e.values.reshape(1, -1))
        y_pred = DT.predict(i2e.values.reshape(1, -1))
        hit = 1-distance.hamming(y_true,y_pred)

        #print(f'MARLENA-{self.neigh_type}\ndecision rule: {rule}\nrule length: {len_rule}\nblack-box decision: {y_true}\nexplained decision: {y_pred}\nfidelity of DT: {fidelity}\nhit: {hit}')
        #print(f'MARLENA-{self.neigh_type}\ndecision rule: {rule}\nrule length: {len_rule}\nblack-box decision: {y_true}\nfidelity of DT: {fidelity}\nhit: {hit}')
        return rule, instance_imporant_feat, fidelity, hit, DT