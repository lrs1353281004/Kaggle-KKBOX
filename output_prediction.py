# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:52:40 2017
get output preditions for the test data
@author: Li Ruosong
"""
print ("Train test and validation sets")

train_label = train['target'].values        
train = train.drop(['target'], axis=1)




ids = test['id'].values
test = test.drop(['id'], axis=1)

for col in train.columns:  
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

d_train=lgb.Dataset(train,train_label)
d_val=lgb.Dataset(train.iloc[5000000:],train_label[5000000:])
#transfer features into category type

    


params = {
        'objective': 'binary',
        
        'boosting': 'gbdt',
        'learning_rate':0.22,
        'lamda_l2': 0.0005,
        'verbose': 0,
        'num_leaves': 3000,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 5000,
        'max_depth': 30,
        'num_rounds': 200,
        'metric' : 'auc'
    }

#%time 
model_f1 = lgb.train(params, train_set=d_train, valid_sets=d_val,verbose_eval=1)

m1_feature_importance=pd.DataFrame(model_f1.feature_importance(),index=train.columns)

params = {
        'objective': 'binary',
        
        'boosting': 'dart',
        'learning_rate':0.22,
        'lamda_l2': 0.0005,
        'verbose': 0,
        'num_leaves': 800,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 50,
        'max_depth': 25,
        'num_rounds': 200,
        'metric' : 'auc'
    }

#%time 
model_f2 = lgb.train(params, train_set=d_train, valid_sets=d_val, verbose_eval=5)
m2_feature_importance=pd.DataFrame(model_f2.feature_importance(),index=train.columns)
###
'''
params = {
        'objective': 'binary',
        
        'boosting': 'rf',
        'learning_rate':0.22,
        'lamda_l2': 0.0005,
        'verbose': 0,
        'num_leaves': 800,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.85,
        'feature_fraction_seed': 1,
        'max_bin': 50,
        'max_depth': 25,
        'num_rounds': 150,
        'metric' : 'auc'
    }

#%time 
model_f3 = lgb.train(params, train_set=d_train,valid_sets=d_val, verbose_eval=5)
m3_feature_importance=pd.DataFrame(model_f3.feature_importance(),index=train.columns)
###
params = {
        'objective': 'binary',
        
        'boosting': 'goss',
        'learning_rate':0.15,
        'lamda_l2': 0.0005,
        'verbose': 0,
        'num_leaves': 800,
        
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 50,
        'max_depth': 25,
        'num_rounds': 200,
        'metric' : 'auc'
    }

#%time 
model_f4 = lgb.train(params, train_set=d_train,valid_sets=d_val, verbose_eval=5)
m4_feature_importance=pd.DataFrame(model_f4.feature_importance(),index=train.columns)
print('Making predictions')
'''
p_test_1 = model_f1.predict(test)

p_test_2 = model_f2.predict(test)
'''
p_test_3 = model_f3.predict(X_test)
p_test_4 = model_f4.predict(X_test)
'''
p_test_avg = np.mean([p_test_1, p_test_2], axis = 0)


print('Done making predictions')

print ('Saving predictions Model model of gbdt')

subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test_avg
subm.to_csv(data_path + 'submission_lgb_1.3.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')

print('Done!')
