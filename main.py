import configparser
from functools import partial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV
import graphics

config = configparser.ConfigParser()
config.read('config.ini')

with open('Report/X.csv', 'r') as file:
    predictors = file.read().splitlines()

with open('Report/X_only_microbiom.csv', 'r') as file:
    predictors_only_microbiom = file.read().splitlines()

with open('Report/X_without_CGM.csv', 'r') as file:
    predictors_without_CGM = file.read().splitlines()

with open('Report/Y.csv', 'r') as file:
    targets = file.read().splitlines()

with open('Report/train_num.csv', 'r') as file:
    train_num = list(map(int, file.read().splitlines()))

with open('Report/test_num.csv', 'r') as file:
    test_num = list(map(int, file.read().splitlines()))

data = pd.read_excel(config["MODEL"]["meals_param"])
train = data[data['N'].isin(train_num)][['N', *predictors, *targets]]
test = data[data['N'].isin(test_num)][['N', *predictors, *targets]]

impute = KNNImputer(n_neighbors=3, weights='distance')

select = SelectPercentile(score_func=partial(mutual_info_regression,
                                             random_state=0),
                          percentile=10)

scale = StandardScaler()

LSVR = Pipeline(steps=[('impute', impute), ('KBest', select), ('preprocess', scale),
                       ('model', LinearSVR(random_state=0,
                                           dual=False,
                                           loss='squared_epsilon_insensitive',
                                           fit_intercept=True,
                                           intercept_scaling=1.0,
                                           max_iter=1000,
                                           tol=1e-05,
                                           verbose=0))])

LSVR0 = Pipeline(steps=[('impute', impute), ('preprocess', scale),
                        ('model', LinearSVR(random_state=0,
                                            dual=False,
                                            loss='squared_epsilon_insensitive',
                                            fit_intercept=True,
                                            intercept_scaling=1.0,
                                            max_iter=1000,
                                            tol=1e-05,
                                            verbose=0,
                                            C=1.0,
                                            epsilon=0.0))])

regressor = GridSearchCV(estimator=LSVR,
                         param_grid=[{'model__epsilon': np.arange(0.0, 0.2),
                                      'model__C': np.arange(1e-5, 2)}],
                         scoring='r2',
                         cv=5)

regressor0 = GridSearchCV(estimator=LSVR0,
                          param_grid=[{}],
                          scoring='r2',
                          cv=5)

for i in targets:
    # Выбираем только те строчки, где в таргете нет пропусков
    stripped_train = train[train[i].notna()]
    # Разбиваем массив на входные и одну выходную переменную
    train_x = stripped_train[predictors]
    train_y = stripped_train.pop(i)

    stripped_test = test[test[i].notna()]
    test_x = stripped_test[predictors]
    test_y = stripped_test.pop(i)

    if i == "BG60":

        # C микробиомом
        print('BG60_model')
        print(f"Число пациентов: {stripped_train['N'].unique().shape[0]}")
        print(f'Число приемов пищи: {stripped_train.shape[0]}')
        regressor.fit(train_x, train_y)
        BG60_features = regressor.best_estimator_['KBest'].get_feature_names_out(input_features=predictors)
        BG60_scores = np.sort(regressor.best_estimator_['KBest'].scores_ /
                              np.max(regressor.best_estimator_['KBest'].scores_))[::-1][:len(BG60_features)]
        graphics.plot_features(BG60_features, BG60_scores, 'BG60 features')
        print(f'Best CV model: \n {regressor.best_params_}')
        print(f'Best CV R2: \n {regressor.best_score_}')
        BG60_predicted_y = regressor.predict(test_x)
        BG60_model_R = pearsonr(test_y, BG60_predicted_y)
        print(f"Test R:{BG60_model_R.statistic}, {BG60_model_R.pvalue}")
        graphics.plot_scatter(test_y, BG60_predicted_y, BG60_model_R.statistic, 'BG60 model')
        print(f'Test MAE: {mean_absolute_error(test_y, BG60_predicted_y)}')
        print(f'Test MSE: {mean_squared_error(test_y, BG60_predicted_y)}')

        # Исключили микробиом
        print('BG60_model without microbiom')
        BG60_features_stripped = []
        for j in BG60_features:
            if j not in predictors_only_microbiom:
                BG60_features_stripped.append(j)
        regressor0.fit(train_x[BG60_features_stripped], train_y)
        print(f'Best CV R2: \n {regressor0.best_score_}')
        BG60_predicted_y0 = regressor0.predict(test_x[BG60_features_stripped])
        BG60_model_R0 = pearsonr(test_y, BG60_predicted_y0)
        print(f"Test R:{BG60_model_R0.statistic}, {BG60_model_R0.pvalue}")
        graphics.plot_scatter(test_y, BG60_predicted_y0, BG60_model_R0.statistic, 'BG60 model without microbiom')
        print(f'Test MAE: {mean_absolute_error(test_y, BG60_predicted_y0)}')
        print(f'Test MSE: {mean_squared_error(test_y, BG60_predicted_y0)}')

    elif i == "BG120":

        # С микробиомом
        print('BG120_model')
        print(f"Число пациентов: {stripped_train['N'].unique().shape[0]}")
        print(f'Число приемов пищи: {stripped_train.shape[0]}')
        regressor.fit(train_x, train_y)
        BG120_features = regressor.best_estimator_['KBest'].get_feature_names_out(input_features=predictors)
        BG120_scores = np.sort(regressor.best_estimator_['KBest'].scores_ /
                               np.max(regressor.best_estimator_['KBest'].scores_))[::-1][:len(BG120_features)]
        graphics.plot_features(BG120_features, BG120_scores, 'BG120 features')
        print(f'Best CV model: \n {regressor.best_params_}')
        print(f'Best CV R2: \n {regressor.best_score_}')
        BG120_predicted_y = regressor.predict(test_x)
        BG120_model_R = pearsonr(test_y, BG120_predicted_y)
        print(f"Test R:{BG120_model_R.statistic}, {BG120_model_R.pvalue}")
        graphics.plot_scatter(test_y, BG120_predicted_y, BG120_model_R.statistic, 'BG120 model')
        print(f'Test MAE: {mean_absolute_error(test_y, BG120_predicted_y)}')
        print(f'Test MSE: {mean_squared_error(test_y, BG120_predicted_y)}')

        # Исключили микробиом
        print('BG120_model without microbiom')
        BG120_features_stripped = []
        for j in BG120_features:
            if j not in predictors_only_microbiom:
                BG120_features_stripped.append(j)
        regressor0.fit(train_x[BG120_features_stripped], train_y)
        print(f'Best CV R2: \n {regressor0.best_score_}')
        BG120_predicted_y0 = regressor0.predict(test_x[BG120_features_stripped])
        BG120_model_R0 = pearsonr(test_y, BG120_predicted_y0)
        print(f"Test R:{BG120_model_R0.statistic}, {BG120_model_R0.pvalue}")
        graphics.plot_scatter(test_y, BG120_predicted_y0, BG120_model_R0.statistic, 'BG120 model without microbiom')
        print(f'Test MAE: {mean_absolute_error(test_y, BG120_predicted_y0)}')
        print(f'Test MSE: {mean_squared_error(test_y, BG120_predicted_y0)}')

    elif i == "BGMax":

        # С микробиомом
        print('BGMax_model')
        print(f"Число пациентов: {stripped_train['N'].unique().shape[0]}")
        print(f'Число приемов пищи: {stripped_train.shape[0]}')
        regressor.fit(train_x, train_y)
        BGMax_features = regressor.best_estimator_['KBest'].get_feature_names_out(input_features=predictors)
        BGMax_scores = np.sort(regressor.best_estimator_['KBest'].scores_ /
                               np.max(regressor.best_estimator_['KBest'].scores_))[::-1][:len(BGMax_features)]
        graphics.plot_features(BGMax_features, BGMax_scores, 'BGMax features')
        print(f'Best CV model: \n {regressor.best_params_}')
        print(f'Best CV R2: \n {regressor.best_score_}')
        BGMax_predicted_y = regressor.predict(test_x)
        BGMax_model_R = pearsonr(test_y, BGMax_predicted_y)
        print(f"Test R:{BGMax_model_R.statistic}, {BGMax_model_R.pvalue}")
        graphics.plot_scatter(test_y, BGMax_predicted_y, BGMax_model_R.statistic, 'BGMax model')
        print(f'Test MAE: {mean_absolute_error(test_y, BGMax_predicted_y)}')
        print(f'Test MSE: {mean_squared_error(test_y, BGMax_predicted_y)}')

        # Исключили микробиом
        print('BGMax_model without microbiom')
        BGMax_features_stripped = []
        for j in BGMax_features:
            if j not in predictors_only_microbiom:
                BGMax_features_stripped.append(j)
        regressor0.fit(train_x[BGMax_features_stripped], train_y)
        print(f'Best CV R2: \n {regressor0.best_score_}')
        BGMax_predicted_y0 = regressor0.predict(test_x[BGMax_features_stripped])
        BGMax_model_R0 = pearsonr(test_y, BGMax_predicted_y0)
        print(f"Test R:{BGMax_model_R0.statistic}, {BGMax_model_R0.pvalue}")
        graphics.plot_scatter(test_y, BGMax_predicted_y0, BGMax_model_R0.statistic, 'BGMax model without microbiom')
        print(f'Test MAE: {mean_absolute_error(test_y, BGMax_predicted_y0)}')
        print(f'Test MSE: {mean_squared_error(test_y, BGMax_predicted_y0)}')

    elif i == "AUC120":

        # С микробиомом
        print('AUC120_model')
        print(f"Число пациентов: {stripped_train['N'].unique().shape[0]}")
        print(f'Число приемов пищи: {stripped_train.shape[0]}')
        regressor.fit(train_x, train_y)
        AUC120_features = regressor.best_estimator_['KBest'].get_feature_names_out(input_features=predictors)
        AUC120_scores = np.sort(regressor.best_estimator_['KBest'].scores_ /
                                np.max(regressor.best_estimator_['KBest'].scores_))[::-1][:len(AUC120_features)]
        graphics.plot_features(AUC120_features, AUC120_scores, 'AUC120 features')
        print(f'Best CV model: \n {regressor.best_params_}')
        print(f'Best CV R2: \n {regressor.best_score_}')
        AUC120_predicted_y = regressor.predict(test_x)
        AUC120_model_R = pearsonr(test_y, AUC120_predicted_y)
        print(f"Test R:{AUC120_model_R.statistic}, {AUC120_model_R.pvalue}")
        graphics.plot_scatter(test_y, AUC120_predicted_y, AUC120_model_R.statistic, 'AUC120 model')
        print(f'Test MAE: {mean_absolute_error(test_y, AUC120_predicted_y)}')
        print(f'Test MSE: {mean_squared_error(test_y, AUC120_predicted_y)}')

        # Исключили микробиом
        print('AUC120_model without microbiom')
        AUC120_features_stripped = []
        for j in AUC120_features:
            if j not in predictors_only_microbiom:
                AUC120_features_stripped.append(j)
        regressor0.fit(train_x[AUC120_features_stripped], train_y)
        print(f'Best CV R2: \n {regressor0.best_score_}')
        AUC120_predicted_y0 = regressor0.predict(test_x[AUC120_features_stripped])
        AUC120_model_R0 = pearsonr(test_y, AUC120_predicted_y0)
        print(f"Test R:{AUC120_model_R0.statistic}, {AUC120_model_R0.pvalue}")
        graphics.plot_scatter(test_y, AUC120_predicted_y0, AUC120_model_R0.statistic, 'AUC120 model without microbiom')
        print(f'Test MAE: {mean_absolute_error(test_y, AUC120_predicted_y0)}')
        print(f'Test MSE: {mean_squared_error(test_y, AUC120_predicted_y0)}')

    elif i == "iAUC120":

        # С микробиомом
        print('iAUC120_model')
        print(f"Число пациентов: {stripped_train['N'].unique().shape[0]}")
        print(f'Число приемов пищи: {stripped_train.shape[0]}')
        regressor.fit(train_x, train_y)
        iAUC120_features = regressor.best_estimator_['KBest'].get_feature_names_out(input_features=predictors)
        iAUC120_scores = np.sort(regressor.best_estimator_['KBest'].scores_ /
                                 np.max(regressor.best_estimator_['KBest'].scores_))[::-1][:len(iAUC120_features)]
        graphics.plot_features(iAUC120_features, iAUC120_scores, 'iAUC120 features')
        print(f'Best CV model: \n {regressor.best_params_}')
        print(f'Best CV R2: \n {regressor.best_score_}')
        iAUC120_predicted_y = regressor.predict(test_x)
        iAUC120_model_R = pearsonr(test_y, iAUC120_predicted_y)
        print(f"Test R:{iAUC120_model_R.statistic}, {iAUC120_model_R.pvalue}")
        graphics.plot_scatter(test_y, iAUC120_predicted_y, iAUC120_model_R.statistic, 'iAUC120 model')
        print(f'Test MAE: {mean_absolute_error(test_y, iAUC120_predicted_y)}')
        print(f'Test MSE: {mean_squared_error(test_y, iAUC120_predicted_y)}')

        # Исключили микробиом
        print('iAUC120_model without microbiom')
        iAUC120_features_stripped = []
        for j in iAUC120_features:
            if j not in predictors_only_microbiom:
                iAUC120_features_stripped.append(j)
        regressor0.fit(train_x[iAUC120_features_stripped], train_y)
        print(f'Best CV R2: \n {regressor0.best_score_}')
        iAUC120_predicted_y0 = regressor0.predict(test_x[iAUC120_features_stripped])
        iAUC120_model_R0 = pearsonr(test_y, iAUC120_predicted_y0)
        print(f"Test R:{iAUC120_model_R0.statistic}, {iAUC120_model_R0.pvalue}")
        graphics.plot_scatter(test_y, iAUC120_predicted_y0, iAUC120_model_R0.statistic,
                              'iAUC120 model without microbiom')
        print(f'Test MAE: {mean_absolute_error(test_y, iAUC120_predicted_y0)}')
        print(f'Test MSE: {mean_squared_error(test_y, iAUC120_predicted_y0)}')

plt.show()
