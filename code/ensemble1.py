from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging_model = BaggingClassifier(DecisionTreeClassifier(),
                                  n_estimators=50, # 모델50개
                                  max_samples=100, # 한 데이터세트 100개
                                  n_jobs=-1,
                                  random_state=42)
bagging_model

# n_estimators: 
# max_samples

# bagging_model.fit(X_train, y_train)

# Bagging을 지원하는 사이킷런 패키지가 존재
from sklearn.ensemble import RandomForestClassifier

fr_model = RandomForestClassifier(n_estimators=50,
                                  max_leaf_nodes=16,
                                  n_jobs=-1, random_state=42)

fr_model

import numpy as np
import matplotlib.pyplot as plt
p = np.arange(0, 1.01, 0.01)
log_odds = np.log(p / (1- p))
plt.plot(p, log_odds)