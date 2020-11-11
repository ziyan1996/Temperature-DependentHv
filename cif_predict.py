#inputs: the cif you want to predict, temperature, load
from dscribe.descriptors import SOAP
from ase.io import read
from ase import Atoms
import torch as th
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import pymatgen as mg
import os
import argparse

parser = argparse.ArgumentParser(description='Predicting hardness of structures at given temperature and applied load')
parser.add_argument('--data_file', required=True, help="Cif file to predict")
parser.add_argument('--composition', required=True, help="Chemical formula")
#parser.add_argument('--temp', required=True, help="At which temperature (K)")
#parser.add_argument('--load', required=True, help="At which applied temperature (N)")
args = parser.parse_args()

path = '/Users/ziyanzhang/Downloads/dscribe/examples/hv_temp_cif/'
dir = os.listdir(path)
structure_files = []
for file in dir:
    structure_files.append(file)

os.chdir(path)
atoms = [None]*591
for i in range(591):
    atoms[i] = read(str(i) + ".cif")
print('got structures from cif files')

df = pd.read_excel('/Users/ziyanzhang/Downloads/dscribe/examples/hv_temp_cif_labels.xlsx')
y = df['hardness'] #lower case y is the target hardness
temp = df['temp']
load = df['load']
print('got targets from spread sheet')

species = set()
for i in range(len(atoms)):
    species.update(atoms[i].get_chemical_symbols())

soap = SOAP(species=species, periodic=True, rcut=5, nmax=1, lmax=1, average="outer")
print('generating SOAP descriptors...')
soap.get_number_of_features()

feature_vectors = soap.create(atoms, n_jobs=1)
feature_tensor = th.tensor(feature_vectors)
print('DONE, SOAP descriptors ready to use')
feature_pd = pd.DataFrame(feature_vectors) #feature pd is the soap descriptors

#generate compositional descriptors
df = pd.read_excel('/Users/ziyanzhang/Downloads/dscribe/examples/hv_temp_cif_labels.xlsx')
class Vectorize_Formula:

    def __init__(self):
        elem_dict = pd.read_excel('/Users/ziyanzhang/Desktop/subgroup/elementsnew.xlsx') # CHECK NAME OF FILE
        self.element_df = pd.DataFrame(elem_dict)
        self.element_df.set_index('Symbol',inplace=True)
        self.column_names = []
        for string in ['avg','diff','max','min']:
            for column_name in list(self.element_df.columns.values):
                self.column_names.append(string+'_'+column_name)

    def get_features(self, formula):
        try:
            fractional_composition = mg.Composition(formula).fractional_composition.as_dict()
            element_composition = mg.Composition(formula).element_composition.as_dict()
            avg_feature = np.zeros(len(self.element_df.iloc[0]))
            std_feature = np.zeros(len(self.element_df.iloc[0]))
            for key in fractional_composition:
                try:
                    avg_feature += self.element_df.loc[key].values * fractional_composition[key]
                    diff_feature = self.element_df.loc[list(fractional_composition.keys())].max()-self.element_df.loc[list(fractional_composition.keys())].min()
                except Exception as e:
                    print('The element:', key, 'from formula', formula,'is not currently supported in our database')
                    return np.array([np.nan]*len(self.element_df.iloc[0])*4)
            max_feature = self.element_df.loc[list(fractional_composition.keys())].max()
            min_feature = self.element_df.loc[list(fractional_composition.keys())].min()
            std_feature=self.element_df.loc[list(fractional_composition.keys())].std(ddof=0)

            features = pd.DataFrame(np.concatenate([avg_feature, diff_feature, np.array(max_feature), np.array(min_feature)]))
            features = np.concatenate([avg_feature, diff_feature, np.array(max_feature), np.array(min_feature)])
            return features.transpose()
        except:
            print('There was an error with the Formula: '+ formula + ', this is a general exception with an unkown error')
            return [np.nan]*len(self.element_df.iloc[0])*4
gf=Vectorize_Formula()

# empty list for storage of features
features=[]

# add values to list using for loop
for formula in df['composition']:
    features.append(gf.get_features(formula))

# feature vectors and targets as X and y
X = pd.DataFrame(features, columns = gf.column_names)
pd.set_option('display.max_columns', None)
# allows for the export of data to excel file
header=gf.column_names
header.insert(0,"Composition")

composition=df['composition']
#composition=pd.read_excel('pred_hv_comp.xlsx',sheet_name='Sheet1', usecols="A")
composition=pd.DataFrame(composition)

predicted=np.column_stack((composition,X))
predicted=pd.DataFrame(predicted) #predicted(dataframe) is the compositional features

array = predicted.values
X = array[:,1:141]
Xpd = pd.DataFrame(X)
Xpd['temp'] = temp
Xpd['load'] = load
scaler = preprocessing.StandardScaler().fit(Xpd)
Xpd = scaler.transform(Xpd)
Xpd = pd.DataFrame(Xpd)
combine_feature_pd = pd.concat([Xpd, feature_pd], axis=1, sort=False) #this is soap+compositional+temp+load

array = combine_feature_pd.values
X = array[:,1:4049]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.99, test_size=0.01,random_state=100, shuffle=True)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree =0.7, learning_rate = 0.18,
                max_depth = 4, alpha = 5, n_estimators = 500, subsample=0.6)
xgb_model=xg_reg.fit(X_train, y_train)
preds=xgb_model.predict(X_test)
r2=r2_score(preds, y_test)
mae=mean_absolute_error(preds, y_test)
#print('R2 score on test set: ' + str(r2))
#print('MAE on test set: ' + str(mae))
print('Model ready to use')

predict_cif=read(args.data_file)
predict_temp=input("at which temperature (K):")
predict_load=input("at which applied load (N):")

predict_soap = soap.create(predict_cif, n_jobs=1)
predsoap = pd.DataFrame(predict_soap)
predict_comp=args.composition

complist=[predict_comp]
comp_pd=pd.DataFrame(complist, columns=['composition'])
gf=Vectorize_Formula()
# empty list for storage of features
pred_features=[]
for form in comp_pd['composition']:
    pred_features.append(gf.get_features(form))
#pred_features.append(gf.get_features(predict_comp))

# add values to list using for loop
#for formula in complist:
#    pred_features.append(gf.get_features(formula))

# feature vectors and targets as X and y

predcomp_fea = pd.DataFrame(pred_features, columns = gf.column_names)
pd.set_option('display.max_columns', None)

# allows for the export of data to excel file
#header=gf.column_names
#header.insert(0,"Composition")

#composition=df['composition']
#composition=pd.read_excel('pred_hv_comp.xlsx',sheet_name='Sheet1', usecols="A")
#predcomp=pd.DataFrame(complist)

#predcomp_fea=np.column_stack((predcomp,predX))
#predcomp_fea=pd.DataFrame(predcomp_fea) #predicted(dataframe) is the compositional features

predcomp_fea['temp']=predict_temp
predcomp_fea['load']=predict_load

#scaler = preprocessing.StandardScaler().fit(predcomp_fea)
predcomp_fea=scaler.transform(predcomp_fea)
predcomp_fea = pd.DataFrame(predcomp_fea)

combine_feature_pred = pd.concat([predcomp_fea, predsoap], axis=1, sort=False)

pred_array = combine_feature_pred.values
X_predict = pred_array[:,1:4049]

preds=xgb_model.predict(X_predict)

print('Predicted hardness:' + str(preds))
