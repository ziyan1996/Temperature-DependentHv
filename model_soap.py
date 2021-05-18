from dscribe.descriptors import SOAP
from ase.io import read
from ase import Atoms
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
import pandas as pd
import numpy as np
import pickle as pk
import os


path = '/Users/ziyanzhang/Downloads/dscribe/examples/updated_cif/'
os.listdir(path)
#load hardness dataset
trainingset_x = pk.load(open('trainingset_x.p', 'rb'))
trainingset_y = pk.load(open('trainingset_y.p', 'rb'))

print('Training dataset loaded')

y = trainingset_y['hardness']
temp = trainingset_y['temp']
load = trainingset_y['load']

species = set()
for i in range(len(trainingset_x)):
    species.update(trainingset_x[str(i)].get_chemical_symbols())

soap = SOAP(species=species, periodic=True, rcut=2.5, nmax=6, lmax=6, average="inner", sparse=False)
print('Generating descriptors...')
atoms=[]
for i in range(len(trainingset_x)):
    atoms.append(trainingset_x[str(i)])
feature_vectors = soap.create(atoms, n_jobs=1)
#feature_tensor = th.tensor(feature_vectors)
print('Descriptors ready to use')

feature_pd = pd.DataFrame(feature_vectors)
feature_pd['temp'] = temp
feature_pd['load'] = load

from sklearn import preprocessing

st_scaler = preprocessing.StandardScaler().fit(feature_pd[["temp", "load"]])
feature_pd[["temp", "load"]] = st_scaler.transform(feature_pd[["temp", "load"]])

from scipy.sparse import csr_matrix

csr_feature_pd = csr_matrix(feature_pd)

from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(csr_feature_pd, y,
#                                                    train_size=0.9, test_size=0.1,
#                                                    random_state=90, shuffle=True)

X_train = csr_feature_pd
y_train = y

import xgboost as xgb

#simply define the regressor, didn't do any training
#xgb_reg = xgb.XGBRegressor(max_depth=4, learning_rate=0.3, n_estimators=500,
#                             verbosity=1, objective='reg:squarederror',
#                             booster='gbtree', tree_method='auto', n_jobs=1,
#                             gamma=0.0001, min_child_weight=8,max_delta_step=0,
#                             subsample=1, colsample_bytree=0.8, colsample_bynode=0.8,
#                             colsample_bylevel=0.8, reg_alpha=0,
#                             reg_lambda=4, scale_pos_weight=1, base_score=0.6,
#                             missing=None, num_parallel_tree=1, importance_type='gain',
#                             eval_metric='rmse',nthread=4)

xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree =0.7, colsample_bynode=0.8,
                           learning_rate = 0.07, colsample_bylevel=0.8,
                           max_depth = 4, subsample=1, n_estimators = 1000).fit(X_train,y_train)

#input the crystal structure file you want to predict
c1 = read('/Users/ziyanzhang/Downloads/dscribe/examples/anycrystalstructure.cif')

c1s = [c1,c1,c1,c1,c1,c1,c1,c1,c1,c1,c1]
c1s_feature_vectors = soap.create(c1s, n_jobs=1)
c1s_feature_pd = pd.DataFrame(c1s_feature_vectors)

c1s_feature_pd['temp'] = [293.15,393.15,493.15,593.15,693.15,793.15,893.15,993.15,1093.15,1193.15,1293.15]
c1s_feature_pd['load'] = 0.49
c1s_feature_pd[["temp", "load"]] = st_scaler.transform(c1s_feature_pd[["temp", "load"]])

c1s_csr_feature_pd = csr_matrix(c1s_feature_pd)

#xgb_model = xgb_reg.fit(X_train, y_train)
#pred = xgb_model.predict(c1s_csr_feature_pd)
#print(pred)

n_repeat = 5
estimator=BaggingRegressor(base_estimator=xgb_reg, max_samples=500, n_estimators=1)
print('bagging estimator assigned')

predict_y1 = np.zeros((len(c1s), n_repeat))

for i in range(n_repeat):
    estimator.fit(X_train, y_train)
    print('one estimator fit')
    predict_y1[:,i] = estimator.predict(c1s_csr_feature_pd)

pred_y1 = pd.DataFrame(predict_y1)
pred_y1.to_excel('pred1_y.xlsx')
