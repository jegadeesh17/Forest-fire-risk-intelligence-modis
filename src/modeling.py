from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def split_data(df, features, target='high_risk'):
    """Split data into training and testing sets."""
    X = df[features]
    y = df[target]
    
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

def train_random_forest(X_train, y_train, n_estimators=100):
    """Train a Random Forest Classifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """Train an XGBoost Classifier."""
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model
