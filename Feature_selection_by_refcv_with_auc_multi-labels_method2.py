
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
import time

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import SCORERS
from sklearn.preprocessing import label_binarize
from sklearn.sklearn.preprocessing import scale

__version__ = '0.1.20200517'

'''
refcv 接收的 y 的shape 要求是 （n, 1），所以可以先用多分类模型计算分类，然后在计算多分类 ROC时，对预测结果进行 binary 处理，之后计算多分类的ROC

相比 method1，这种方式计算的 ROC 偏小一些
'''

def roc_auc_with_multi_labels(estimator, X, y, n_classes=3, weight='micro',
                     mapping={'Normal':[1, 0, 0], 'I_III': [0, 1, 0], 'IV': [0, 0, 1]}):
    y_test = np.array(list(map(lambda x: mapping[x], y)))
    y_score = estimator.predict_proba(X)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    if weight == 'micro':
        return roc_auc["micro"]
    elif weight == 'macro':
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        return roc_auc["macro"]
    else:
        raise ValueError('"weight" is not in ("macro", "micro")


def plot_roc_cv(Scores):
    Scores_arr = np.array(Scores)
    n_fea = Scores_arr.shape[1]
    fig, ax = plt.subplots()

    ax.fill_between(range(1, Scores_arr.shape[1]+1), Scores_arr.max(axis=0), Scores_arr.min(axis=0), color='skyblue')
    ax.plot(range(1, n_fea+1), Scores_arr.max(axis=0), '-*', lw=2, color='orange', label='max')
    ax.plot(range(1, n_fea+1), Scores_arr.min(axis=0), '-*', lw=2, color='blue', label='min')
    ax.plot(range(1, n_fea+1), Scores_arr.mean(axis=0), '--', lw=1, color='green', label='mean')
    ax.plot(range(1, n_fea+1), Scores_arr.mean(axis=0) + Scores_arr.std(axis=0), '--', lw=1.5,color='darksalmon', label='mean + std')
    ax.plot(range(1, n_fea+1), Scores_arr.mean(axis=0) - Scores_arr.std(axis=0), '--', lw=1.5,color='darkviolet', label='mean - std')
    ax.plot(range(1, n_fea+1), [max(Scores_arr.max(axis=0))]* n_fea, '--', lw=1,color='lawngreen')
    ax.plot(range(1, n_fea+1), [min(Scores_arr.min(axis=0))]* n_fea, '--', lw=1,color='lawngreen')
    plt.legend(loc='upper right', fontsize=7)
    plt.ylim([0, 1])
    plt.xlim([1, 9])
    plt.xlabel('feature num')
    plt.ylabel('AUC')
    yticks = [_/10 for _ in  range(0, 11, 2)]
    yticks.append(round(max(Scores_arr.max(axis=0)), 3))
    yticks.append(round(min(Scores_arr.min(axis=0)), 3))
    yticks = sorted(yticks)
    plt.yticks(yticks, yticks)

    plt.title('Feature importance selected by RFECV \n with RF(100 tree)')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('feature_label.csv', sep='\t')
    X = df[df.columns[:-2]]
    y = df['Transfer'].values    
    N = 100
    Scores = []
    Best_fea_num = []
    Ranks = []
    t0 = time.time()
    for i in range(1, N+1):
        X1, y1 = shuffle(X, y, random_state=i)
        model = RandomForestClassifier(n_estimators=100)
        rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(10),
                  n_jobs=-1, scoring=roc_auc_with_multi_labels)
        rfecv.fit(X1, y1)
        Best_fea_num.append(rfecv.n_features_)
        Scores.append(rfecv.grid_scores_)
        Ranks.append(rfecv.ranking_)
        if i % 10 == 0:
            t1 = time.time()
            print('finish {} round test in {} sec'.format(i, t1-t0))
            t0 = t1
   plot_roc_cv(Scores)