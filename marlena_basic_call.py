from marlena.marlena import MARLENA
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import json

bb = joblib.load('./black_boxes/RandomForest.pkl') 
numerical_vars = np.load('./data/numerical_vars.npy').tolist()
labels_name = np.load('./data/labels_name.npy').tolist()
categorical_vars = np.load('./data/categorical_vars.npy').tolist()
df = pd.read_csv('./data/preprocessed_patient-characteristics.csv')
m1 = MARLENA(neigh_type='unified')
i2e = df.loc[3, numerical_vars+categorical_vars]
rule, instance_imporant_feat, fidelity, hit, DT = m1.extract_explanation(i2e, df, bb, numerical_vars, categorical_vars, labels_name, k=200, size=2000,alpha=0.1)


res = {"rule": rule, 'feat_importance': instance_imporant_feat, 'fidelity': fidelity, 'hit': hit}

print(json.dumps(res))