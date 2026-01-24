from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import confusion_matrix,precision_score,precision_recall_curve,classification_report
from sklearn.metrics import recall_score,f1_score,fbeta_score,roc_auc_score,average_precision_score


def load_breast_cancer_scaled(test_size=0.2, random_state=42):
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Train/test split
    x_train, x_test, t_train, t_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, t_train, t_test



def sigmoid(logit):
    return 1 / (1 + np.exp(-logit))


def cost_f(X, t, weights,eps = 1e-15):
    examples = X.shape[0]
    pred = np.dot(X, weights)
    logit = sigmoid(pred)
    logit = np.clip(logit,eps,1-eps)
    error = np.dot(t , np.log(logit)) + np.dot((1 - t) , np.log(1 - logit))
    cost = -error / examples
    return cost

# (p-t)*xn
def f_dervative(X, t, weights,eps = 1e-15):
    examples = X.shape[0]
    pred = np.dot(X, weights)
    logit = sigmoid(pred)
    logit = np.clip(logit,eps,1-eps)
    error = logit - t
    gradient = X.T @ error
    return gradient

# Focal Loss : FL(p,y)=−αy(1−p)^γlog(p)−(1−α)(1−y)p^γlog(1−p)
def compute_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Compute focal loss for binary classification problem.

    y_true: array-like, true labels; shape should be (n_samples,)
    y_pred: array-like, predicted probabilities for being in the
        positive class; shape should be (n_samples,)
    alpha: float, weight for the positive class to improve further
    gamma: float, focusing parameter for focal loss
    """

    # Ensure the prediction is within the range [eps, 1-eps] to avoid log(0)
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1.0 - eps)

    # Compute focal loss
    pos_loss = -alpha * np.power(1 - y_pred, gamma) * np.log(y_pred)
    neg_loss = -(1 - alpha) * np.power(y_pred, gamma) * np.log(1 - y_pred)

    # Combine losses
    focal_loss = np.where(y_true == 1, pos_loss, neg_loss)

    return np.mean(focal_loss)



def gradient_descent_logistic_regression(X, t, step_size = 0.01, precision = 0.0001, max_iter = 500):
    examples, features = X.shape
    iter = 0
    cur_weights = np.random.rand(features)         # random starting point
    last_weights = cur_weights + 100 * precision    # something different

    print(f'Initial Random Cost: {cost_f(X, t, cur_weights)}')

    while norm(cur_weights - last_weights) > precision and iter < max_iter:
        last_weights = cur_weights.copy()           # must copy
        gradient = f_dervative(X, t, cur_weights)
        cur_weights -= gradient * step_size
        #print(cost_f(X, cur_weights))
        iter += 1

    print(f'Total Iterations {iter}')
    print(f'Optimal Cost: {cost_f(X, t, cur_weights)}')
    return cur_weights

def predict(X,cur_weights,threshold=0.5):
    logit = np.dot(X,cur_weights)
    prob = sigmoid(logit)
    pred = np.where(prob>=threshold,1,0)
    return pred , prob

def accuracy_score(t,pred):
    return sum(t==pred) / t.size

def report_accuracy(y_true,y_pred):
    tn = sum((y_true == 0) & (y_pred == 0))
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    print(f'confusion matrix :\ntn:{tn} , fp:{fp}\nfn:{fn} , tp:{tp}')
    print(f'balanced accuracy: {1/2 * (tp / ( tp + fn ) +  tn / (tn + fp))}')
    Precision = tp/(tp+fp)
    Recall = tp/(tp+fn)
    print(f'Precision: {Precision}')
    print(f'Recall : {Recall}')
    f1_score = (2 * Precision * Recall) / (Precision + Recall) 
    print(f'F1 Score : {f1_score}') 

if __name__ == '__main__':
    #np.random.seed(0)  # If you want to fix the results

    x_train, x_test, t_train, t_test = load_breast_cancer_scaled()
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
    cur_weights = gradient_descent_logistic_regression(x_train,t_train)
    pred_t,prob = predict(x_test,cur_weights,threshold=0.45)
    report_accuracy(t_test,pred_t)
    print(f'Micro Metric: {f1_score(t_test,pred_t,average='micro')}')
    print(f'Macro Metric: {f1_score(t_test,pred_t,average='macro')}')
    print(f'Weighted Macro Metric: {f1_score(t_test,pred_t,average='weighted')}')
    print(f"{classification_report(t_test,pred_t)}")
    precisions,recalls,thresholds = precision_recall_curve(t_test,prob)
    precisions,recalls = precisions[:-1],recalls[:-1]
    plt.plot(thresholds ,precisions , 'b--' , label = 'precision' )
    plt.plot(thresholds ,recalls , 'r--' , label = 'recall' )
    plt.legend()
    plt.show()
    # print('roc_auc_score', roc_auc_score(t_test, prob))        # Can be optimistic on imbalanced data
    # print('average precision score', average_precision_score(t_test, prob))  # Better for imbalanced data
