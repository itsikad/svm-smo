# Sequential Minimal Optimization
A vanila numpy implementation of SVM using
Sequential Minimal Optimization algorithm.

A One-vs-ALL strategy is employed for multiclass classification task.

Original paper:
[John Platt. Sequential minimal optimization: A fast algorithm for training support vector machines. In _Technical Report MSR-TR-98-14, Microsoft Research, 1998a_](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf)

# Setup

```bash
git clone https://github.com/itsikad/svm-smo.git
cd svm-smo
pip install -r requirements.txt
```

# Init / Train / Predict

```python
from smo_optimizer import SVM

# dataset
x_train/test = ...
y_train/test = ...  # binary labels

# init model
model = SVM(kernel_type='rbf')

# train
model.fit(x_train, y_train)

# predict
y_pred = model.predict(x_test)
```

# Run example code
Example uses Iris Flower dataset. 
Employs a One-vs-All strategy (OneVsAllClassifier) to solve a multi-class classification problem.

```
python example.py
```