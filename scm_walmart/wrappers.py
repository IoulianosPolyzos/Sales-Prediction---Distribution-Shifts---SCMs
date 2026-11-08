import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor, LGBMClassifier


class LGBMRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, parents, cat_features_list, lgb_params):
        self.parents = sorted(parents)
        self.cat_features_list = cat_features_list
        self.lgb_params = lgb_params
        self.model = LGBMRegressor(**(lgb_params or {}))
        self.encoders = {}

    def fit(self, X, y):
        X_enc = np.zeros(X.shape, dtype=float)
        cat_indices = []

        for i, p in enumerate(self.parents):
            if p in self.cat_features_list:
                cat_indices.append(i)
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                # Κάνουμε fit_transform και το αποθηκεύουμε για το predict
                col_data = X[:, i].astype(str).reshape(-1, 1)
                X_enc[:, i] = enc.fit_transform(col_data).flatten()
                self.encoders[i] = enc
            else:
                X_enc[:, i] = X[:, i].astype(float)

        if cat_indices:
            self.model.fit(X_enc, y, categorical_feature=cat_indices)
        else:
            self.model.fit(X_enc, y)
        return self

    def predict(self, X):
        X_enc = np.zeros(X.shape, dtype=float)
        for i, p in enumerate(self.parents):
            if p in self.cat_features_list:
                col_data = X[:, i].astype(str).reshape(-1, 1)
                X_enc[:, i] = self.encoders[i].transform(col_data).flatten()
            else:
                X_enc[:, i] = X[:, i].astype(float)

        return self.model.predict(X_enc)


class LGBMClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, parents, cat_features_list, lgb_params):
        self.parents = sorted(parents)
        self.cat_features_list = cat_features_list
        self.lgb_params = lgb_params
        self.model = LGBMClassifier(**(lgb_params or {}))
        self.encoders = {}

    def fit(self, X, y):
        X_enc = np.zeros(X.shape, dtype=float)
        cat_indices = []

        for i, p in enumerate(self.parents):
            if p in self.cat_features_list:
                cat_indices.append(i)
                enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                col_data = X[:, i].astype(str).reshape(-1, 1)
                X_enc[:, i] = enc.fit_transform(col_data).flatten()
                self.encoders[i] = enc
            else:
                X_enc[:, i] = X[:, i].astype(float)

        if cat_indices:
            self.model.fit(X_enc, y, categorical_feature=cat_indices)
        else:
            self.model.fit(X_enc, y)
        return self

    def predict(self, X):
        X_enc = np.zeros(X.shape, dtype=float)
        for i, p in enumerate(self.parents):
            if p in self.cat_features_list:
                col_data = X[:, i].astype(str).reshape(-1, 1)
                X_enc[:, i] = self.encoders[i].transform(col_data).flatten()
            else:
                X_enc[:, i] = X[:, i].astype(float)
        return self.model.predict(X_enc)

    def predict_proba(self, X):
        X_enc = np.zeros(X.shape, dtype=float)
        for i, p in enumerate(self.parents):
            if p in self.cat_features_list:
                col_data = X[:, i].astype(str).reshape(-1, 1)
                X_enc[:, i] = self.encoders[i].transform(col_data).flatten()
            else:
                X_enc[:, i] = X[:, i].astype(float)
        return self.model.predict_proba(X_enc)

    @property
    def classes_(self):
        return self.model.classes_