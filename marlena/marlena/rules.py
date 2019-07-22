import numpy as np

def istance_rule_extractor(i2e_values, DT, features_names, labels_names, categorical_vars, numerical_vars):

    """
    :param i2e_values: np.array, shape=(1, -1) containing values of that instance features
    :param DT: pre-trained decision tree from sklearn
    :param features_names:  list of features names
    :return: a rule (str) describing why that instance was classified in that way by the DT and rule lenght (int)
    """
    
    #n_nodes = DT.tree_.node_count
    #print('numero di nodi nel tree: '+str(n_nodes))
    #children_left = DT.tree_.children_left
    #children_right = DT.tree_.children_right
    feature = DT.tree_.feature
    threshold = DT.tree_.threshold
    
    #estraggo il path di nodi seguiti per arrivare alla foglia che contiene il mio esempio
    node_indicator = DT.decision_path(i2e_values)
    #print('path:')
    #print(node_indicator)
    #node indicator contiene una matice con sulla prima colonna la tupla (id_sample,node_id) in questo caso è il path di un 
    #solo esempio quindi id_sample=0, invece node_id contiente tutti i nodi utilizzati da quella istanza


    #trovo l'id del nodo che è la foglia dove cade il mio esempio
    leave_id = DT.apply(i2e_values)
    #leave_id è una vettore al cui posto i-esimo si trova l'id del nodo-foglia in cui cade l'esempio i-esimo 
    #in questo caso l'esempio è solo uno, quindi è un'array lunga 1
    #print('id leaf node: '+ str(leave_id[0]))

    #qui trovo in nodi usati
    node_index = node_indicator.indices
    
    #ho solo un sample, quindi il suo id è per forza 0
    sample_id = 0

    #salvo le split conditions in una lista
    istance_string = list()
    list_split_conditions = list()

    for node_id in node_index:
        # controllo che non siamo già in una foglia
        if leave_id[sample_id] == node_id:
            # \print("leaf node {} reached, no decision here".format(leave_id[sample_id]))
            break

        if features_names[feature[node_id]] in categorical_vars:
            istance_string.append(f'{features_names[feature[node_id]]} = {bool(i2e_values[0][feature[node_id]])}')
            list_split_conditions.append(
                f'{features_names[feature[node_id]]} = {bool(i2e_values[0][feature[node_id]])}')
        else:
            istance_string.append(f'{features_names[feature[node_id]]} = {i2e_values[0][feature[node_id]]}\n')
            # print(str(features_names[feature[node_id]])+'='+str(round(i2e_values[0][feature[node_id]],2))+threshold_sign+str(round(threshold[node_id],2)))
            # se il valore di quella feature in quella istanza è minore della treshold
            if i2e_values[0][feature[node_id]] <= threshold[node_id]:
                threshold_sign = " <= "
            else:
                threshold_sign = " > "
            list_split_conditions.append(f'{features_names[feature[node_id]]}{threshold_sign}{round(threshold[node_id], 2)}')
    instance_imporant_feat = '\n'.join(istance_string)

    mask = DT.predict(i2e_values)[0].astype(bool)
    decision_rule = '{' + ',\n '.join(list_split_conditions) + '} -> ' + str(np.array(labels_names)[mask])
    return decision_rule, instance_imporant_feat ,len(list_split_conditions)