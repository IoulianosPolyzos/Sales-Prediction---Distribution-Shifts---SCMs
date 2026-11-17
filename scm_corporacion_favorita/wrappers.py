import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor, LGBMClassifier


class LGBMRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, parents=None, cat_features_list=None, lgb_params=None):
        self.parents = parents
        self.cat_features_list = cat_features_list
        self.lgb_params = lgb_params

    def fit(self, X, y):
        # Create model here to avoid pickling/cloning issues with sub-objects
        self.model = LGBMRegressor(**(self.lgb_params or {}))
        self.encoders = {}

        # Sort parents during fit, not __init__
        actual_parents = sorted(self.parents) if self.parents else []
        cat_features = self.cat_features_list if self.cat_features_list else []

        X_enc = np.zeros(X.shape, dtype=float)
        cat_indices = []

        for i, p in enumerate(actual_parents):
            if p in cat_features:
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
        actual_parents = sorted(self.parents) if self.parents else []
        cat_features = self.cat_features_list if self.cat_features_list else []

        X_enc = np.zeros(X.shape, dtype=float)
        for i, p in enumerate(actual_parents):
            if p in cat_features:
                col_data = X[:, i].astype(str).reshape(-1, 1)
                X_enc[:, i] = self.encoders[i].transform(col_data).flatten()
            else:
                X_enc[:, i] = X[:, i].astype(float)

        return self.model.predict(X_enc)


class LGBMClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, parents=None, cat_features_list=None, lgb_params=None):
        self.parents = parents
        self.cat_features_list = cat_features_list
        self.lgb_params = lgb_params

    def fit(self, X, y):
        self.model = LGBMClassifier(**(self.lgb_params or {}))
        self.encoders = {}

        actual_parents = sorted(self.parents) if self.parents else []
        cat_features = self.cat_features_list if self.cat_features_list else []

        X_enc = np.zeros(X.shape, dtype=float)
        cat_indices = []

        for i, p in enumerate(actual_parents):
            if p in cat_features:
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
        actual_parents = sorted(self.parents) if self.parents else []
        cat_features = self.cat_features_list if self.cat_features_list else []

        X_enc = np.zeros(X.shape, dtype=float)
        for i, p in enumerate(actual_parents):
            if p in cat_features:
                col_data = X[:, i].astype(str).reshape(-1, 1)
                X_enc[:, i] = self.encoders[i].transform(col_data).flatten()
            else:
                X_enc[:, i] = X[:, i].astype(float)
        return self.model.predict(X_enc)

    def predict_proba(self, X):
        actual_parents = sorted(self.parents) if self.parents else []
        cat_features = self.cat_features_list if self.cat_features_list else []

        X_enc = np.zeros(X.shape, dtype=float)
        for i, p in enumerate(actual_parents):
            if p in cat_features:
                col_data = X[:, i].astype(str).reshape(-1, 1)
                X_enc[:, i] = self.encoders[i].transform(col_data).flatten()
            else:
                X_enc[:, i] = X[:, i].astype(float)
        return self.model.predict_proba(X_enc)

    @property
    def classes_(self):
        return self.model.classes_