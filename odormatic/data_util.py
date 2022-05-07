## This could be cleaned up a bit. Only need one or two actual results from all the datastructures.

import numpy as np
import csv
import copy
import sklearn.cross_decomposition as cross
from sklearn import linear_model as lm
from sklearn import model_selection as mod_sel
from scipy.stats import t, norm
from . import mol_utils as mu
import sys

import time

import string


def dependent_corr(
    xy,
    xz,
    yz,
    n,
    twotailed=True,
):
    """
    Calculates the statistic significance between two dependent correlation
    coefficients. Is \rho_{XY} equal to \rho_{XZ}?
    H0: xy==xz
    @param xy: correlation coefficient between x and y
    @param xz: correlation coefficient between x and z
    @param yz: correlation coefficient between y and z
    @param n: number of elements in x, y and z
    @param twotailed: whether to calculate a one or two tailed test
    @return: t and p-val
    """
    d = xy - xz
    determin = 1 - xy * xy - xz * xz - yz * yz + 2 * xy * xz * yz
    av = (xy + xz) / 2
    cube = (1 - yz) * (1 - yz) * (1 - yz)

    t2 = d * np.sqrt((n - 1) * (1 + yz) /
                     (((2 * (n - 1) / (n - 3)) * determin + av * av * cube)))
    p = 1 - t.cdf(abs(t2), n - 3)

    if twotailed:
        p *= 2

    return t2, p


def get_correlation(get_embedding_fn_all_words,
                    do_contextual=False,
                    return_score_by_descriptor=False):
    # Load the Dream and Dravnieks ratings from csv files into lists

    Dravnieks, DRV_mols, DRV_words = mu.load_mols_csv(
        'data/Dravnieks_perception.csv', first_col=1, mol_col=0)
    Dream, DRM_mols, DRM_words = mu.load_mols_csv(
                                                  'data/AvgOdorFeatures.csv',
                                                  first_col=4)
    # There were molecules that were excluded in the original data files, so they were
    # added by making these parallel Dream/Dravnieks ratings matrices (which are later
    # appended to the originals)
    Dream2, DRM_mols2, _ = mu.load_mols_csv('data/AvgOdorFeatures2.csv',
                                            first_col=4)

    # Preprocess descriptor labels (e.g., replace multi-word terms with single-word equivalents)
    DRM_words, DRV_words = mu.preprocess(DRM_words, DRV_words,
                                         do_contextual)

    #Collect indices of descriptors that couldn't be found in dictionary
    remove_inds_DRV = [i for i, w in enumerate(DRV_words) if w == '---']

    # Load Distributional Semantic Model (word embeddings)
    model = get_embedding_fn_all_words(DRV_words + DRM_words)
    model.pop('---', None)

    #Collect indices of descriptors that couldn't be found in dictionary for later removal
    for i, w in enumerate(DRV_words):
        assert w.lower() == w
        if i not in remove_inds_DRV:
            try:
                model[w.lower().strip()]
            except:
                raise LookupError("Could not find word {} in model".format(w))

    #Remove all words not in dictionary
    DRV_words = [
        w for i, w in enumerate(DRV_words) if i not in remove_inds_DRV
    ]

    # Make the lists of the Dream and Dravnieks molecules correspond to each other.
    # Remove molecules from the Dravnieks list that are not in the Dream list, and vice versa.
    mols = [i for i in DRV_mols if i in DRM_mols]
    mols2 = [i for i in DRV_mols if i in DRM_mols2]

    # print(f"drv_words={DRV_words}")
    # print(f"drm_words={DRM_words}")

    # Ensure that the embeddings matrices only have word vectors for descriptors included in the
    # final, pared descriptor lists
    Sx = np.array([model[w.lower()] for w in DRM_words])
    Sy = np.array([model[w.lower()] for w in DRV_words])

    # Make lists containing the indices of the molecules removed from the Dravnieks and Dream lists
    # for future reference.
    remove_mols_DRM = [i for i, w in enumerate(DRM_mols) if w not in DRV_mols]
    remove_mols_DRV = [i for i, w in enumerate(DRV_mols) if w not in DRM_mols]

    remove_mols_DRM2 = [
        i for i, w in enumerate(DRM_mols2) if w not in DRV_mols
    ]
    remove_mols_DRV2 = [
        i for i, w in enumerate(DRV_mols) if w not in DRM_mols2
    ]

    # Remove columns/rows for deleted molecules/descriptors from matrices
    Px2, Py2 = mu.clean_matrices(Dream, Dravnieks, remove_inds_DRV,
                                 remove_mols_DRM, remove_mols_DRV)
    Pxx2, Pyy2 = mu.clean_matrices(Dream2, Dravnieks, remove_inds_DRV,
                                   remove_mols_DRM2, remove_mols_DRV2)

    # Append the ratings for the new and original molecule lists
    Px2 = np.vstack((Px2, Pxx2))
    Py2 = np.vstack((Py2, Pyy2))
    mols = mols + mols2

    # Initialize MultiTask Elastic Net with cross-validation for setting parameter weights
    Reg = cross.PLSRegression  #lm.MultiTaskElasticNetCV #cross.PLSCanonical #cross.PLSRegression

    # Normalize semantic vector matrices
    Sxx = copy.copy(Sx)
    Syy = copy.copy(Sy)
    Sxx_mean = np.mean(Sxx.T, 0)
    Syy_mean = np.mean(Syy.T, 0)
    Sxx = (Sxx.T - Sxx_mean).T
    Syy = (Syy.T - Syy_mean).T
    Sxx2 = Sxx / np.linalg.norm(Sxx, axis=1)[:, np.newaxis]
    Syy2 = Syy / np.linalg.norm(Syy, axis=1)[:, np.newaxis]

    # Initialize and fit semantics-only model for transforming from dream to dravnieks
    # modelX2 = Reg(cv=10,max_iter=1e4,fit_intercept=False)
    modelX2 = Reg(n_components=19)
    ThetaX2 = modelX2.fit(Sxx2.T, Syy2.T).coef_

    # Create dicts to store the results
    medians = {}  #store median correlations across molecules
    mediansPvals = {}  #store median Z-scores across molecules above baseline
    sqmeans = {}
    mediansSqErrReductions = {}
    meansSqErrReductions = {}
    sqerrs = {}
    corrs = {}  #store the actual correlations

    #these keys correspond to the different ways to predict the ratings
    keys = ['Semantics2', 'Baseline']

    # Populate dicts with keys for each prediction method
    for key in keys:
        medians[key] = {}
        mediansPvals[key] = {}
        corrs[key] = {}
        sqerrs[key] = {}
        sqmeans[key] = {}
        meansSqErrReductions[key] = {}
        mediansSqErrReductions[key] = {}

    ################################################################################################
    # Measure performance of model that does not use any molecular perceptual rating information   #
    ################################################################################################
    test_size = 1.0
    for key in keys:
        corrs[key][test_size] = []
        medians[key][test_size] = []
        mediansPvals[key][test_size] = []
        sqmeans[key][test_size] = []
        sqerrs[key][test_size] = []
        mediansSqErrReductions[key][test_size] = []
        meansSqErrReductions[key][test_size] = []

    key = 'Baseline'
    for i in range(Py2.shape[0]):
        # Baseline model just assigns a constant rating to each descriptor-molecule combo in the case where there are zero training molecules
        corrs[key][test_size].append(0)
        sqerrs[key][test_size].append(np.linalg.norm(Py2[i, :])**2)

    # Generate predictions for Semantics and Half models (they are the same in this case are there are no training ratings)
    hat = modelX2.predict(Px2)

    key = 'Semantics2'

    if return_score_by_descriptor:
        for i in range(Py2.shape[1]):
            corrs[key][test_size].append(mu.corrcoef(hat[:, i], Py2[:, i]))
        return corrs[key][test_size], DRV_words

    else:
        for i in range(Py2.shape[0]):
            corrs[key][test_size].append(mu.corrcoef(hat[i, :], Py2[i, :]))
            sqerrs[key][test_size].append(
                np.linalg.norm(hat[i, :] - Py2[i, :])**2)
        sqmeans[key][test_size].append(mu.sqmean(corrs[key][test_size]))
        mediansSqErrReductions[key][test_size].append(
            np.median([
                jj / kk for jj, kk in zip(sqerrs[key][test_size],
                                          sqerrs['Baseline'][test_size])
            ]))
        meansSqErrReductions[key][test_size].append(
            np.mean([
                jj / kk for jj, kk in zip(sqerrs[key][test_size],
                                          sqerrs['Baseline'][test_size])
            ]))
        medians[key][test_size].append(np.median(corrs[key][test_size]))
        mediansPvals[key][test_size].append(
            np.median([
                mu.nanreplace(dependent_corr(jj,
                                             0,
                                             0,
                                             hat[i, :].size,
                                             twotailed=False)[1],
                              diff=jj) for jj in corrs[key][test_size]
            ]))

        assert len(medians[key][test_size]) == 1
        return medians[key][test_size][0]
