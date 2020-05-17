
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


class RandomForestClassifierRefcv(RandomForestClassifier):
    '''
    使用 Refcv 进行循环特征删除，选出最佳的特征组合， Refcv 接受的 y 的 shape 要是 (n, 1)。
    binary 处理后，将特征处理成： ['a1,b1,c1',
                        'a2,b2,c2']
   因此在训练前，需要“解码”成正常的格式：[a1, b1, c1
                               a2,b2, c2]   
    '''
    def fit(self, X, y,*args, **kwargs):
        y = np.array([list(map(int, _.split(','))) for _ in y])
        super(RandomForestClassifierRefcv, self).fit(X, y, *args, **kwargs)

def label_binarize_for_Refcv(y_binary):
    '''
    使用 Refcv 进行循环特征删除，选出最佳的特征组合， Refcv 接受的 y 的 shape 要是 (n, 1)。
    对于多分类问题，正常 binary 处理后，y 的shape 是 (n, n_classes)，这是不符合要求的。因此，将其转换成 ['a1,b1,c1',
                                                                       'a2,b2,c2']
    '''
    y = list(map(lambda x:'{},{},{}'.format(*x), y_binary))
    return y

def roc_auc_with_multi_labels(estimator, X, y, n_classes=3, weight='micro'):
    '''
    ROC 原始定义是适用于二分类，如何可以参考
    https://www.jianshu.com/p/00ef5b63dfc8
    '''
    y_pred = estimator.predict_proba(X)
    y_pred_ = np.array([_[:, 1] for _ in y_pred]).T
    y_test = np.array([list(map(int, _.split(','))) for _ in y])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(y_test.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area（方法二）
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_.ravel())
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
        y1 = label_binarize(y, classes=['Normal', 'I_III', 'IV'])
        X1, y1 = shuffle(X, y1, random_state=i)
        y1 = list(map(lambda x:'{},{},{}'.format(*x), y1))
        model = RandomForestClassifierRefcv(n_estimators=100)
        rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(6),
                  n_jobs=-1, scoring=roc_auc_with_multi_labels_binary)
        rfecv.fit(X1, y1)
        Best_fea_num.append(rfecv.n_features_)
        Scores.append(rfecv.grid_scores_)
        Ranks.append(rfecv.ranking_)
        if i % 10 == 0:
            t1 = time.time()
            print('finish {} round test in {} sec'.format(i, t1-t0))
            t0 = t1
   plot_roc_cv(Scores)