import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

from src.smo_optimizer import SVM, OneVsAllClassifier


# Load and process dataset
iris_dataset = load_iris()

x_train, x_test, y_train, y_test = train_test_split(
    iris_dataset.data,
    iris_dataset.target,
    shuffle=True,
    test_size=0.3,
    stratify=iris_dataset.target
)

print(f'Dataset split summary:')
print(f'Training set size: {x_train.shape[0]}')
print(f'Testing set size: {x_test.shape[0]}')

# Visualize training dataset
iris_df = pd.DataFrame(x_train, columns=iris_dataset.feature_names)
iris_df['species'] = iris_dataset.target_names[y_train.reshape(-1,1)]
sns.pairplot(iris_df, hue=iris_df.columns[-1])
plt.show()

# Train Multiclass SVM

solver = OneVsAllClassifier(
    solver=SVM,
    num_classes=len(iris_dataset.target_names),
    c=1.0,
    kkt_thr=1e-3,
    max_iter=1e3,
    kernel_type='rbf',
    gamma_rbf=1.
)

solver.fit(x_train, y_train)

# Predict
y_pred = solver.predict(x_test)

# Performance analysis
print(classification_report(y_test, y_pred, target_names=iris_dataset.target_names))