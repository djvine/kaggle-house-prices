#!/usr/bin/env python3

""" house_prices.py Predict house prices in Ames, Ohio"""

__author__ = "David J. Vine"
__email__ = "djvine@gmail.com"


import pandas as pd
import numpy as np
import numpy.linalg as npl
import sklearn.ensemble as se
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
import operator
import ipdb

def fix_missing_data(df):
    # Row 2120 is missing all basement data
    df[df.index in [2120, 2188]].filter(regex='Bsmt', axis=1).columns
    for col in columns:
        if df[col].dtype in ['float', 'int']:
            df.loc[df.index==2120, col] = 0
        elif col=='BsmtCond':
            df.loc[df.index==2120, col] = 'Po'
        elif col=='BsmtExposure':
            df.loc[df.index==2120, col] = 'No'
        elif col=='BsmtFinType1':
            df.loc[df.index==2120, col] = 'Unf'
        elif col=='BsmtFinType2':
            df.loc[df.index==2120, col] = 'Unf'
        elif col=='BsmtQual':
            df.loc[df.index==2120, col] = 'NA'


def main():
    
    train_df = pd.read_csv('./train.csv', header=0)
    test_df = pd.read_csv('./test.csv', header=0)
    df = pd.concat([train_df, test_df])
    df.reset_index(inplace=True)

    cf = pd.DataFrame()

    # First create a Frame with all numerical data

    # Continuous
    cf['1stFlrSF'] = df['1stFlrSF'].fillna(0.0)
    cf['2ndFlrSF'] = df['2ndFlrSF'].fillna(0.0)
    cf['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0.0)
    cf['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(0.0)
    cf['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0.0)
    cf['EnclosedPorch'] = df['EnclosedPorch'].fillna(0.0)
    cf['GarageArea'] = df['GarageArea'].fillna(0.0)
    cf['GrLivArea'] = df['GrLivArea'].fillna(0.0)
    cf['LotArea'] = df['LotArea'].fillna(0.0) # Has outliers
    cf['OpenPorchSF'] = df.OpenPorchSF.fillna(0.0)
    cf['TotalBsmtSF'] = df.TotalBsmtSF.fillna(0.0)
    cf['WoodDeckSF'] = df.WoodDeckSF.fillna(0.0)

    # Discrete
    cf['BedroomAbvGr'] = df['BedroomAbvGr'].astype(int)
    cf['BsmtFullBath'] = df['BsmtFullBath'].fillna(0).astype(int)
    cf['BsmtHalfBath'] = df['BsmtFullBath'].fillna(0).astype(int)
    cf['Fireplaces'] = df['Fireplaces'].astype(int)
    cf['FullBath'] = df.FullBath.astype(int)
    cf['GarageCars'] = df.GarageCars.fillna(2.0).astype(int)
    cf['GarageYrBlt'] = df.GarageYrBlt.fillna(df.YearBuilt).astype(int)
    cf['HalfBath'] = (df.HalfBath > 0).map({True: 1, False: 0}).astype(int)
    cf['KitchenAbvGr'] = df.KitchenAbvGr.astype(int)
    cf['MoSold'] = df.MoSold.astype(int)
    cf['TotRmsAbvGrd'] = df.TotRmsAbvGrd.astype(int)
    cf['YearBuilt'] = df.YearBuilt
    cf['YearRemodAdd'] = df.YearRemodAdd
    cf['YrSold'] = df.YrSold

    # Nominal
    cf['Alley'] = df.Alley.isnull().map({True: 0, False: 1}) # 93%
    cf['BldgType'] = pd.factorize(df['BldgType'])[0]
    cf['CentralAir'] = df.CentralAir.map({'Y': 1, 'N': 0})
    cf['Condition1'] = pd.factorize(df.Condition1)[0]
    cf['Condition2'] = (df.Condition2 == 'Norm').map({True: 1, False: 0})
    cf['Electrical'] = (df.Electrical.fillna('SBrkr') == 'Sbrkr').map({True: 1, False: 0})
    cf['Exterior1st'] = pd.factorize(df.Exterior1st.fillna(df.Exterior1st.mode()[0]))[0]
    cf['Exterior2nd'] = pd.factorize(df.Exterior2nd.fillna(df.Exterior2nd.mode()[0]))[0]
    cf['Fence'] = pd.factorize(df.Fence.fillna(0))[0]
    cf['Foundation'] = pd.factorize(df.Foundation)[0]
    cf['Functional'] = (df.Functional == 'Typ').map({True: 1, False: 0})
    cf['GarageFinish'] = pd.factorize(df.GarageFinish.fillna(0))[0]
    cf['GarageType'] = pd.factorize(df.GarageType.fillna(0))[0]
    cf['GasAirFurnace'] = (df.Heating == 'GasA').map({True: 1, False: 0})
    cf['HouseStyle'] = pd.factorize(df.HouseStyle)[0]
    cf['LandContour'] = (df.LandContour == 'Lvl').map({True: 1, False: 0})
    cf['LandSlope'] = (df.LandSlope == 'Gtl').map({True: 1, False: 0})
    cf['LotConfig'] = pd.factorize(df.LotConfig)[0]
    cf['LotShape'] = (df.LotShape.isin(['Reg', 'IR1'])).map({True: 1, False: 0})
    cf['MSSubClass'] = pd.factorize(df.MSSubClass)[0]
    cf['MSZoning'] = pd.factorize(df.MSZoning.fillna('RL'))[0]
    cf['MasVnrType'] = pd.factorize(df.MasVnrType.fillna(0))[0]
    cf['Neighborhood'] = pd.factorize(df.Neighborhood)[0]
    cf['PavedDrive'] = (df.PavedDrive=='Y').map({True: 1, False: 0})
    cf['RoofStyle'] = pd.factorize(df.RoofStyle)[0]
    cf['SaleCondition'] = pd.factorize(df.SaleCondition)[0]
    cf['SaleType'] = pd.factorize(df.SaleType.fillna(value='WD'))[0]

    # Ordinal
    CONDITION_MAP = { 'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    SIMPLIFIED_CONDITION_MAP = { 'NA': 0, 'Po': 0, 'Fa': 1, 'TA': 1, 'Gd': 2, 'Ex': 2}
    EXPOSURE_MAP = {'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    SIMPLIFIED_EXPOSURE_MAP = {'NA': 0, 'No': 0, 'Mn': 1, 'Av': 1, 'Gd': 2}
    BSMT_FINTYPE_MAP = {'NA': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    SIMPLIFIED_BSMT_FINTYPE_MAP = {'NA': 0, 'Unf': 0, 'LwQ': 1, 'Rec': 1, 'BLQ': 2, 'ALQ': 2, 'GLQ': 2}
    SIMPLIFIED_OVERALL = {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3}
    cf['BsmtCond'] = df['BsmtCond'].fillna('NA').map(CONDITION_MAP).astype(int)
    cf['Simplified_BsmtCond'] = df['BsmtCond'].fillna('NA').map(SIMPLIFIED_CONDITION_MAP).astype(int)
    cf['BsmtExposure'] = df['BsmtExposure'].fillna('NA').map(EXPOSURE_MAP).astype(int)
    cf['Simplified_BsmtExposure'] = df['BsmtExposure'].fillna('NA').map(SIMPLIFIED_EXPOSURE_MAP).astype(int)
    cf['BsmtFinType1'] = df['BsmtFinType1'].fillna('NA').map(BSMT_FINTYPE_MAP).astype(int)
    cf['Simplified_BsmtFinType1'] = df['BsmtFinType1'].fillna('NA').map(SIMPLIFIED_BSMT_FINTYPE_MAP).astype(int)
    cf['BsmtFinType2'] = df['BsmtFinType2'].fillna('NA').map(BSMT_FINTYPE_MAP).astype(int)
    cf['Simplified_BsmtFinType2'] = df['BsmtFinType2'].fillna('NA').map(SIMPLIFIED_BSMT_FINTYPE_MAP).astype(int)
    cf['BsmtQual'] = df['BsmtQual'].fillna('NA').map(CONDITION_MAP).astype(int)
    cf['Simplified_BsmtQual'] = df['BsmtQual'].fillna('NA').map(SIMPLIFIED_CONDITION_MAP).astype(int)
    cf['ExterCond'] = df['ExterCond'].map(CONDITION_MAP).astype(int)
    cf['Simplified_ExterCond'] = df['ExterCond'].map(SIMPLIFIED_CONDITION_MAP).astype(int)
    cf['ExterQual'] = df['ExterQual'].map(CONDITION_MAP).astype(int)
    cf['Simplified_ExterQual'] = df['ExterQual'].map(SIMPLIFIED_CONDITION_MAP).astype(int)
    cf['FireplaceQu'] = df['FireplaceQu'].fillna('NA').map(CONDITION_MAP).astype(int)
    cf['Simplified_FireplaceQu'] = df['FireplaceQu'].fillna('NA').map(SIMPLIFIED_CONDITION_MAP).astype(int)
    cf['GarageCond'] = df.GarageCond.fillna('NA').map(CONDITION_MAP).astype(int)
    cf['Simplified_GarageCond'] = df.GarageCond.fillna('NA').map(SIMPLIFIED_CONDITION_MAP).astype(int)
    cf['GarageQual'] = df.GarageQual.fillna('NA').map(CONDITION_MAP).astype(int)
    cf['Simplified_GarageQual'] = df.GarageQual.fillna('NA').map(SIMPLIFIED_CONDITION_MAP).astype(int)
    cf['HeatingQC'] = df.HeatingQC.map(CONDITION_MAP).astype(int)
    cf['Simplified_HeatingQC'] = df.HeatingQC.map(SIMPLIFIED_CONDITION_MAP).astype(int)
    cf['KitchenQual'] = df.KitchenQual.fillna('NA').map(CONDITION_MAP).astype(int)
    cf['Simplified_KitchenQual'] = df.KitchenQual.fillna('NA').map(SIMPLIFIED_CONDITION_MAP).astype(int)
    cf['OverallCond'] = df.OverallCond.astype(int)
    cf['Simplified_OverallCond'] = df.OverallCond.map(SIMPLIFIED_OVERALL)
    cf['OverallQual'] = df.OverallQual.astype(int)
    cf['Simplified_OverallQual'] = df.OverallQual.map(SIMPLIFIED_OVERALL)

    # Impute
    cf['LotFrontage'] = df['LotFrontage']
    rfr = se.RandomForestRegressor(n_estimators=200, n_jobs=-1)
    y = cf.loc[cf.LotFrontage.notnull(), 'LotFrontage']
    X = cf.loc[cf.LotFrontage.notnull(), :].drop('LotFrontage', axis=1)
    if False:
        score = cross_val_score(rfr, X.values, y.values)
        print('{:s} impute accuracy: {:0.2f}%'.format('LotFrontage', 100.0*score.mean()))
    rfr.fit(X, y)
    cf.loc[cf.LotFrontage.isnull(), 'LotFrontage'] = rfr.predict(
        cf.loc[cf.LotFrontage.isnull(), :].drop('LotFrontage', axis=1).values)

    # Feature Engineering
    cf['Has_Alley'] = df.Alley.isnull().map({True: 0, False: 1}) # 93%
    cf['Is_1Fam'] = (df.BldgType == '1Fam').map({True: 1, False: 0}) # 83%
    cf['Damaged'] = (df.Functional.isin(['Typ', 'Min1', 'Min2', 'Mod'])).map({True: 0, False: 1})
    cf['Has_Garage'] = (df.GarageType.isnull()).map({True: 1, False: 0})
    cf['Has1Kitchen'] = (df.KitchenAbvGr == 1).map({True: 1, False: 0})
    cf['Has_LowQualFinSF'] = (df.LowQualFinSF>0).map({True: 1, False: 0})
    cf['Has_1story'] = (df.MSSubClass.isin([20, 30, 40, 120])).map({True: 1, False: 0})
    cf['Has_MasVnrArea'] = (df.MasVnrArea>0).map({True: 1, False: 0})
    cf['Has_shed'] = (df.MiscFeature == 'Shed').map({True: 1, False: 0})
    cf['Has_Valuable_Feature'] = (df.MiscVal>=2000).map({True: 1, False: 0})
    cf['SeasonSold'] = df.MoSold.map({
        1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4})
    cf['Has_OpenPorch'] = (df.OpenPorchSF>0).map({True: 1, False: 0})
    cf['Simplified_Overall_Condition'] = (df.OverallCond>5).map({True: 1, False: 0})
    cf['Simplified_Overall_Quality'] = (df.OverallQual>5).map({True: 1, False: 0})
    cf['Has_Pool'] = (df.PoolQC.notnull()).map({True: 1, False: 0})
    cf['Has_Shingle_Roof'] = (df.RoofMatl=='CompShg').map({True: 1, False: 0})
    cf['Has_Screen_Porch'] = (df.ScreenPorch>0).map({True: 1, False: 0})
    cf['Has_Paved_Street'] = (df.Street=='Pave').map({True: 1, False: 0})
    cf['Has_Wood_Deck'] = (df.WoodDeckSF>0).map({True: 1, False: 0})
    cf['Built_Recently'] = (df.YearBuilt>1990).map({True: 1, False: 0})
    cf['Has_Remodel'] = (df.YearBuilt != df.YearRemodAdd).map({True: 1, False: 0})
    cf['Age'] = 2010-df.YearBuilt
    cf['Time_Since_Sold'] = 2010-df.YrSold
    cf['Remodel_Age'] = 2010-df.YearRemodAdd

    # Total square feet
    area_cols = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'EnclosedPorch', 'GarageArea', 'GrLivArea', 'LotFrontage', 'LotArea', 'LowQualFinSF', 'MasVnrArea',
    'OpenPorchSF', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'WoodDeckSF']

    cf['Total_Area'] = df[area_cols].sum(axis=1)

    # Neighborhood quality metric
    avg_price_by_hood = df.loc[
        df.SalePrice.notnull(), ['SalePrice',
        'Neighborhood']].groupby('Neighborhood').mean().sort_values(by='SalePrice')
    cf['Neighborhood_Value'] = df.Neighborhood.map(
        avg_price_by_hood.to_dict()['SalePrice']).apply(lambda x: np.log(x))

    # Unskew numeric features.
    numeric_features = cf.dtypes[cf.dtypes != 'object'].index
    from scipy.stats import skew
    skewness = cf[numeric_features].apply(lambda x: skew(x))
    skewed = cf.columns[skewness>0.75]
    cf[skewed] = np.log1p(cf[skewed])

    # Scale the data
    from sklearn.preprocessing import StandardScaler, RobustScaler
    Q1 = cf.quantile(0.25)
    Q3 = cf.quantile(0.75)
    IQR = Q3 - Q1
    has_outliers = ((cf < ( Q1-1.5*IQR )) | (cf > ( Q3+1.5*IQR ))).any()
    cols_w_outliers = cf.columns[has_outliers]
    cols_wo_outliers = cf.columns[~has_outliers]
    sscaler = StandardScaler()
    rscaler = RobustScaler()
    s_scaled = sscaler.fit_transform(cf[cols_wo_outliers])
    r_scaled = rscaler.fit_transform(cf[cols_w_outliers])
    for i, feature in enumerate(cols_wo_outliers):
        cf[feature] = s_scaled[:,i]
    for i, feature in enumerate(cols_w_outliers):
        cf[feature] = r_scaled[:, i]

    if True: # Plot correlation matrix
        import seaborn as sns
        corr = cf.corr()
        sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
        

    # Add interaction variables
    if True:
        poly_degree = 2
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(poly_degree, interaction_only=True)
        int_feats = poly.fit_transform(cf[numeric_features])
        for i in range(int_feats.shape[1]):
            cf = pd.concat([cf, pd.Series(int_feats[:,i], name=poly.get_feature_names()[i])], axis=1)
        
        cf_corr = cf.corr(method='spearman')

        # Ignore self
        mask = np.ones(cf_corr.columns.size) - np.eye(cf_corr.columns.size)
        cf_corr *= mask

        drops = []
        for col in cf_corr.columns.values:
            if np.in1d([col], drops):
                continue

            corr = cf_corr[abs(cf_corr[col])>0.98].index
            drops = np.union1d(drops, corr)

        print("Dropping {:d} highly correlated features: {:}".format(len(drops), drops))
        cf.drop(drops, axis=1, inplace=True)

    # Add one-hot encoding
    def encode_one_hot(oh):
        
        encoded_df = pd.DataFrame()
        encoded_df[oh.name] = oh

        dummies = pd.get_dummies(encoded_df[oh.name], prefix = '_'+oh.name)

        encoded_df = encoded_df.join(dummies)
        encoded_df.drop([oh.name], axis=1, inplace=True)

        return encoded_df

    def do_onehot(tf):
        
        # Onehot the nominal categories
        onehot_df = pd.DataFrame(index=tf.index)
        onehot_df = onehot_df.join(encode_one_hot(tf['Alley'].fillna(value='NA')))
        onehot_df = onehot_df.join( encode_one_hot(tf['BldgType']))
        onehot_df = onehot_df.join( encode_one_hot(tf.CentralAir))
        onehot_df = onehot_df.join( encode_one_hot(tf.Condition1))
        onehot_df = onehot_df.join( encode_one_hot((tf.Condition2 == 'Norm')))
        onehot_df = onehot_df.join( encode_one_hot(tf.Electrical.fillna('SBrkr')))
        onehot_df = onehot_df.join( encode_one_hot(tf.Exterior1st.fillna(tf.Exterior1st.mode()[0])))
        onehot_df = onehot_df.join( encode_one_hot(tf.Exterior2nd.fillna(tf.Exterior2nd.mode()[0])))
        onehot_df = onehot_df.join( encode_one_hot(tf.Fence.fillna(0)))
        onehot_df = onehot_df.join( encode_one_hot(tf.Foundation))
        onehot_df = onehot_df.join( encode_one_hot((tf.Functional == 'Typ')))
        onehot_df = onehot_df.join( encode_one_hot((tf.GarageFinish.fillna(0))))
        onehot_df = onehot_df.join( encode_one_hot((tf.GarageType.fillna(0))))
        onehot_df = onehot_df.join( encode_one_hot((tf.Heating == 'GasA')))
        onehot_df = onehot_df.join( encode_one_hot(tf.HouseStyle))
        onehot_df = onehot_df.join( encode_one_hot((tf.LandContour == 'Lvl')))
        onehot_df = onehot_df.join( encode_one_hot((tf.LandSlope == 'Gtl')))
        onehot_df = onehot_df.join( encode_one_hot(tf.LotConfig))
        onehot_df = onehot_df.join( encode_one_hot((tf.LotShape.isin(['Reg', 'IR1']))))
        onehot_df = onehot_df.join( encode_one_hot(tf.MSSubClass))
        onehot_df = onehot_df.join( encode_one_hot(tf.MSZoning.fillna('RL')))
        onehot_df = onehot_df.join( encode_one_hot(tf.MasVnrType.fillna(0)))
        onehot_df = onehot_df.join( encode_one_hot(tf.Neighborhood))
        onehot_df = onehot_df.join( encode_one_hot((tf.PavedDrive=='Y')))
        onehot_df = onehot_df.join( encode_one_hot(tf.RoofStyle))
        onehot_df = onehot_df.join( encode_one_hot(tf.SaleCondition))
        onehot_df = onehot_df.join( encode_one_hot(tf.SaleType.fillna('WD')))

        # Add ordinal features
        if True:
            onehot_df = onehot_df.join(encode_one_hot(tf['BsmtCond'].fillna('NA')))
            onehot_df = onehot_df.join( encode_one_hot(tf['BsmtExposure'].fillna('NA')))
            onehot_df = onehot_df.join( encode_one_hot(tf['BsmtFinType1'].fillna('NA')))
            onehot_df = onehot_df.join( encode_one_hot(tf['BsmtFinType2'].fillna('NA')))
            onehot_df = onehot_df.join( encode_one_hot(tf['BsmtQual'].fillna('NA')))
            onehot_df = onehot_df.join( encode_one_hot(tf['ExterCond']))
            onehot_df = onehot_df.join( encode_one_hot(tf['ExterQual']))
            onehot_df = onehot_df.join( encode_one_hot(tf['FireplaceQu']))
            onehot_df = onehot_df.join( encode_one_hot(tf.GarageCond))
            onehot_df = onehot_df.join( encode_one_hot(tf.GarageQual))
            onehot_df = onehot_df.join( encode_one_hot(tf.HeatingQC))
            onehot_df = onehot_df.join( encode_one_hot(tf.KitchenQual))
            onehot_df = onehot_df.join( encode_one_hot(tf.OverallCond))
            onehot_df = onehot_df.join( encode_one_hot(tf.OverallQual))

        return onehot_df

    df_onehot = do_onehot(df)

    # Drop columns with less than 5 non-zero entries
    #df_onehot.drop(df_onehot.columns[df_onehot.sum(axis=0)<5], axis=1, inplace=True)

    cf = cf.join(df_onehot)

    # Metric depends on log of SalePrice
    y = train_df['SalePrice'].apply(lambda x: np.log(x))
    X = cf.loc[df.SalePrice.notnull()]
    Xhat = cf.loc[df.SalePrice.isnull()]

    print('Train dataset shape: ', X.values.shape)
    print('Test dataset shape: ', Xhat.values.shape)

    if False: # scikit-learn Random Forest
        rfr = se.RandomForestRegressor(n_estimators=200, n_jobs=-1)
        scores = cross_val_score(rfr, X.values, y.values)
        print('Cross validation score: {:2.5f}'.format(np.mean(scores)))

        rfr.fit(X.values, y.values)
        print('RFR on train data: {:2.3f}'.format(100.0*mean_squared_error(y.values, rfr.predict(X.values))))

        output = pd.DataFrame()
        output['Id'] = test_df['Id']
        output['SalePrice'] = np.exp(rfr.predict(Xhat.values))
        output.to_csv('./submission_sklearn_rf.csv', index=False)

    if False: # Normal equation
        mat_x = np.matrix(X.values)
        mat_xhat = np.matrix(Xhat.values)
        mat_y = np.matrix(y.values)
        theta = np.dot(npl.pinv(np.dot(np.transpose(mat_x), mat_x)), np.dot(np.transpose(mat_x), np.transpose(mat_y)))
        # check
        ipdb.set_trace()
        print(np.sum(np.abs(np.array(np.dot(np.transpose(theta), np.transpose(mat_x))-np.transpose(mat_y)))**2))
        print(np.sum(np.abs(np.array(np.dot(mat_x, theta)-np.transpose(mat_y)))**2))
        output = pd.DataFrame()
        output['Id'] = test_df['Id']
        output['SalePrice'] = np.exp(np.array(np.dot(mat_xhat, theta)))
        output.to_csv('./submission_normal.csv', index=False)
    
    if True: # XGBoost 
        import xgboost as xgb
        def report(grid_scores, n_top=5):
            params = None
            top_scores = sorted(grid_scores, key=operator.itemgetter(1), reverse=True)[:n_top]
            for i, score in enumerate(top_scores):
                print("Parameters with rank: {0}".format(i + 1))
                print("Mean validation score: {0:.4f} (std: {1:.4f})".format(
                    score.mean_validation_score, np.std(score.cv_validation_scores)))
                print("Parameters: {0}".format(score.parameters))
                print("")
            if params == None:
                params = score.parameters
            return params

        grid_test = {
            'colsample_bytree': [0.1, 0.2, 0.5],
            'gamma': [0.0, 0.2, 0.4],
            'max_depth': [5, 10, 25],
            'min_child_weight': [0.1, 1.5, 2.5],
            'n_estimators': [1000, 7000, 20000],
            'reg_alpha': [0.1, 0.5, 1.0],
            'reg_lambda': [0.1, 0.5, 1.0],
            'subsample': [0.1, 0.2, 0.3],
        }
        print('Hyperparameter optimisation')
        regr_test = xgb.XGBRegressor(
                     learning_rate=0.05,
                     seed=298729190,
                     silent=1)

        regr = xgb.XGBRegressor(
                     colsample_bytree=0.2,
                     gamma=0.0,
                     learning_rate=0.05,
                     max_depth=6,
                     min_child_weight=1.5,
                     n_estimators=7300,
                     reg_alpha=0.9,
                     reg_lambda=0.5,
                     subsample=0.2,
                     seed=10483,
                     silent=1)

        #grid_search = GridSearchCV(estimator = regr_test, param_grid=grid_test, n_jobs=-1, cv=10)
        #grid_search.fit(X, y)
        #best_params = report(grid_search.grid_scores_)

        regr.fit(X, y)
        print('XGB on test data: {:2.3f}'.format(100.0*mean_squared_error(y.values, regr.predict(X))))

        # Prune unimportance features
        if False:
            ipdb.set_trace()
            _feature_importance = regr.feature_importances_
            _feature_importance *= 100.0/_feature_importance.max()
            importance_threshhold = 15.0
            importance_index = np.where(_feature_importance>importance_threshhold)[0]
            important_features = cf.columns.index[important_features]
            print('There are {:d} important features'.format(important_features.shape[0]))
            sorted_index = np.argsort(feature_importances_[importance_threshhold])[::-1]
            print('Features sorted by importance:')
            print(important_features[sorted_index])

            pos = np.arange(sorted_index.shape[0]) + .5
            plt.subplot(1, 2, 2)
            plt.barh(pos, _feature_importance[importance_index][sorted_index[::-1]], align='center')
            plt.yticks(pos, important_features[sorted_index[::-1]])
            plt.xlabel('Relative Importance')
            plt.title('Variable Importance')
            plt.draw()
            plt.show()

            X = X[:, importance_index][:, sorted_index]
            regr.fit(X, y)
            print('Pruned XGB on test data: {:2.3f}'.format(100.0*mean_squared_error(y.values, regr.predict(X))))


        

        yhat_xgboost = regr.predict(Xhat)
        output = pd.DataFrame()
        output['Id'] = test_df['Id']
        output['SalePrice'] = np.exp(regr.predict(Xhat))
        output.to_csv('./submission_xgboost.csv', index=False)

    if True:
        from sklearn.linear_model import Lasso

        best_alpha = 0.0005
        #params = { 'alpha': np.linspace(0.00001, 0.1, 10000)}
        #regr = Lasso(max_iter=50000)
        #grid_search = GridSearchCV(estimator = regr, param_grid=params, n_jobs=-1, cv=10)
        #grid_search.fit(X, y)
        #best_params = report(grid_search.grid_scores_)

        regr = Lasso(alpha=best_alpha, max_iter=50000)
        regr.fit(X, y)
        print('Lasso on test data: {:2.3f}'.format(100.0*mean_squared_error(y.values, regr.predict(X))))
        yhat_lasso = regr.predict(Xhat)
        output = pd.DataFrame()
        output['Id'] = test_df['Id']
        output['SalePrice'] = np.exp(regr.predict(Xhat))
        output.to_csv('./submission_lasso.csv', index=False)

        output = pd.DataFrame()
        output['Id'] = test_df['Id']
        output['SalePrice'] = np.exp(0.5*(yhat_xgboost+yhat_lasso))
        output.to_csv('./submission_blend.csv', index=False)

if __name__=='__main__':
    main()

