import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_val_score
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from functools import reduce

import warnings
warnings.filterwarnings("ignore")

models = {
    "lasso": Lasso(max_iter=1000000, tol=5e-4),
    "ridge": Ridge(),
    "svr": SVR(),
    "randomforest": RandomForestRegressor()
}

param_grids = {
    "lasso": {'lasso__alpha': [0.01, 0.1, 1, 10]},
    "ridge": {'ridge__alpha': [0.01, 0.1, 1, 10]},
    "svr": {'svr__C': [0.1, 1, 10], 'svr__epsilon': [0.1, 0.2]},
    "randomforest": {'randomforest__n_estimators': [100, 200], 'randomforest__max_depth': [5, 10]}
}

results_list = []
n_bootstrap = 20

df = pd.read_excel('Image_features_chemical_dataset.xlsx')
X = df[df.columns[1:12]].to_numpy()

Chemical = df[df.columns[12:]]
Chemical_name = df.columns[12:]

# Chemical_name = Chemical_name[:3]
# Chemical = Chemical[Chemical_name]

rand_int = [7515, 5938, 1079, 1021, 4641, 1198, 6946, 5208, 6597, 3407]

results_df_ls = []
max_r2_ls = []
max_r2_pos_ls = []

for i in range(10):
    # Set random seed for reproducibility
    np.random.seed(rand_int[0])
    for chemname in Chemical.columns:
        chem = Chemical[chemname].values
        chem = stats.zscore(chem)
        for model_name, model in models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()), 
                (model_name, model)
            ])

            param_grid = param_grids[model_name]

            # Inner 5-fold cross-validation for parameter tuning
            inner_cv_r2 = KFold(n_splits=5, shuffle=True, random_state=9)
            grid_search_r2 = GridSearchCV(pipeline, param_grid, cv=inner_cv_r2, scoring='r2', n_jobs=-1)
            
            inner_cv_rmse = KFold(n_splits=5, shuffle=True, random_state=9)
            grid_search_rmse = GridSearchCV(pipeline, param_grid, cv=inner_cv_rmse, scoring='neg_mean_squared_error', n_jobs=-1)

            # Bootstrap evaluation
            bootstrap_r2 = []
            bootstrap_rmse = []
        
            for _ in range(n_bootstrap):
                    # Resample X and chemical
                    randomstate = np.random.randint(10000)
                    # print(f"Random state: {randomstate}")
                    X_train, X_test, y_train, y_test = train_test_split(X, chem, test_size=0.20, random_state=randomstate)

                    # Inner cross-validation for parameter tuning based on R2 (on the 75% training set)
                    grid_search_r2.fit(X_train, y_train)
                    best_model_r2 = grid_search_r2.best_estimator_
                    best_model_rmse = grid_search_rmse.fit(X_train, y_train).best_estimator_

                    y_pred_r2 = best_model_r2.predict(X_test)
                    y_pred_rmse = best_model_rmse.predict(X_test)
                    
                    # Calculate R2 and RMSE on the test set
                    r2 = r2_score(y_test, y_pred_r2)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred_rmse))

                    bootstrap_r2.append(r2)
                    bootstrap_rmse.append(rmse)

            # Store the results for each model and chemical
            result = {
                'Chemical': chemname,
                'Model': model_name,
                'Mean_R2': np.mean(bootstrap_r2),
                'Std_R2': np.std(bootstrap_r2),
                'Mean_RMSE': np.mean(bootstrap_rmse),
                'Std_RMSE': np.std(bootstrap_rmse)
            }
            results_list.append(result)

            chem_index = list(Chemical.columns).index(chemname)

            print("\r Process{}%".format(round((chem_index+1)*100/len(Chemical.columns))), end="")
            # Print the model is done
            # print(f"Model {model_name} for Chemical {chemname} done, r2: {np.mean(bootstrap_r2)} +- {np.std(bootstrap_r2)}, RMSE: {np.mean(bootstrap_rmse)} +- {np.std(bootstrap_rmse)}.")

    results_df = pd.DataFrame(results_list)
    # results_df.to_csv('model_evaluation_results.csv', index=False)
    results_df_ls.append(results_df)

    # Find the max Mean_R2 of each Chemical
    max_r2 = results_df.loc[results_df.groupby('Chemical')['Mean_R2'].idxmax()][['Chemical', 'Model', 'Mean_R2']]
    # max_r2.to_csv('max_r2.csv')
    max_r2_ls.append(max_r2)

    max_r2_pos = max_r2[max_r2['Mean_R2'] > 0]
    # max_r2_pos.to_csv('max_r2_pos.csv')
    max_r2_pos_ls.append(max_r2_pos)


results_df_all = pd.concat(results_df_ls, axis=0)
results_df_all = results_df_all.reset_index(drop=True, inplace=False)
# results_df_all.to_csv('results_df_all.csv', index=False)
results_df_all = pd.read_csv('results_df_all.csv')

max_r2 = pd.concat(max_r2_ls, axis=0)
max_r2 = max_r2.reset_index(drop=True, inplace=False)
# max_r2.to_csv('max_r2_all.csv', index=False)
max_r2 = pd.read_csv('max_r2_all.csv')

concatenated_chemcial = pd.concat(max_r2_pos_ls, axis=0)
concatenated_chemcial = concatenated_chemcial.reset_index(drop=False, inplace=False)
# concatenated_chemcial.to_csv('concatenated_chemcial_r2.csv', index=False)
concatenated_chemcial = pd.read_csv('concatenated_chemcial_r2.csv')

summary_r2 = concatenated_chemcial.groupby('Chemical').agg(frequency=('Chemical', 'size'), mean_value=('Mean_R2', 'mean')).reset_index()
# summary_r2.to_csv('summary_r2.csv', index=False)
summary_r2 = pd.read_csv('summary_r2.csv')

# sort the summary_r2 by mean_value
summary_r2.sort_values(by='mean_value', ascending=False)
np.unique(concatenated_chemcial['Model'], return_counts=True)