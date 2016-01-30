"""
http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
Taking X and Y two variables of interest and Z the matrix with all the variable minus {X, Y},
the algorithm can be summarized as
    1) perform a normal linear least-squares regression with X as the target and Z as the predictor
    2) calculate the residuals in Step #1
    3) perform a normal linear least-squares regression with Y as the target and Z as the predictor
    4) calculate the residuals in Step #3
    5) calculate the correlation coefficient between the residuals from Steps #2 and #4;
    The result is the partial correlation between X and Y while controlling for the effect of Z
"""

import numpy as np
from scipy import stats, linalg


def partial_corr(C, names, exclude):
    part_corr = {}
    C = np.asarray(C)
    p = C.shape[1]
    for i in range(p):
        if names[i] not in exclude:
            create = True
            for j in range(i + 1, p):
                idx = np.ones(p, dtype=np.bool)
                idx[i] = False
                idx[j] = False
                beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
                beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]
                res_j = C[:, j] - C[:, idx].dot(beta_i)
                res_i = C[:, i] - C[:, idx].dot(beta_j)
                corr = stats.pearsonr(res_i, res_j)[0]
                if abs(corr) > 0.05:
                    if create:
                        part_corr[names[i]] = {}
                        create = False
                    part_corr[names[i]][names[j]] = corr
    return part_corr
