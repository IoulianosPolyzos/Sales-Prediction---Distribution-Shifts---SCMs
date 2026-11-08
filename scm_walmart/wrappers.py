from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from lightgbm import LGBMRegressor, LGBMClassifier


class LGBMRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, parents, cat_features_list, lgb_params):
        self.parents = sorted(parents)
        self.cat_features_list = cat_features_list
        self.lgb_params = lgb_params
        self.model = LGBMRegressor(**(lgb_params or {}))

    def fit(self, X, y):
        cat_indices = [
            i for i, p in enumerate(self.parents)
            if p in self.cat_features_list
        ]

        if cat_indices:
            self.model.fit(X, y, categorical_feature=cat_indices)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class LGBMClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, parents, cat_features_list, lgb_params):
        self.parents = sorted(parents)
        self.cat_features_list = cat_features_list
        self.lgb_params = lgb_params
        self.model = LGBMClassifier(**(lgb_params or {}))

    def fit(self, X, y):
        cat_indices = [
            i for i, p in enumerate(self.parents)
            if p in self.cat_features_list
        ]
        if cat_indices:
            self.model.fit(X, y, categorical_feature=cat_indices)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)