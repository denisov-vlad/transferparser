#!.venv/bin/python3.5

import time
import numpy as np
import itertools
from sklearn.decomposition import PCA
import pprint
from scipy import linalg, stats as stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize, scale, PolynomialFeatures, Imputer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn import datasets, linear_model, manifold
from lib.partial_corr import partial_corr
import matplotlib.pyplot as plt
import matplotlib.cm as cm

pp = pprint.PrettyPrinter()

#разбивает на тестовую и обучаемую выборки
#trn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size=0.4)


class PlayerAnalysis:
    def __init__(self):
        self.data_file = "players_min_15-11-20.csv"
        with open(self.data_file, 'r') as f:
            self.var_names = f.readline().replace('"', '').replace('\n', '').split(';')
        self.player_loaded = np.loadtxt(open(self.data_file, "rb"), delimiter=";", skiprows=1)
        self.shape = self.player_loaded.shape
        self.player_stats = np.delete(self.player_loaded, np.s_[0, 2, 3], axis=1)
        self.player_value = self.player_loaded.T[2].astype(int)
        self.regr_names = []
        self.pca_reduced = None
        self.pca_labels = None
        self.k_means_clusters = None
        self.clf = [linear_model.LinearRegression(),
                    #linear_model.LogisticRegression(), linear_model.Ridge(),
                    #linear_model.BayesianRidge(), linear_model.ARDRegression(), linear_model.TheilSenRegressor(),
                    #linear_model.PassiveAggressiveRegressor(), linear_model.ElasticNet(), linear_model.Lasso()
                    ]
        self.transforms = [self._regr_iden, self._regr_ln, self._regr_logistic, self._regr_obr]
        #print(self.shape)

    def reset_clf(self):
        self.clf = [linear_model.LinearRegression(),
                    #linear_model.LogisticRegression(), linear_model.Ridge(),
                    #linear_model.BayesianRidge(), linear_model.ARDRegression(), linear_model.TheilSenRegressor(),
                    #linear_model.PassiveAggressiveRegressor(), linear_model.ElasticNet(), linear_model.Lasso()
                    ]

    @staticmethod
    def _regr_iden(x):
        return x

    @staticmethod
    def _regr_ln(x):
        return np.log(x+0.01)

    @staticmethod
    def _regr_logistic(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def _regr_obr(x):
        return 1/(x+0.01)


    @staticmethod
    def _regr_func_apply(functions, values):
        temp = np.empty(shape=(19, 0))
        for func, val in zip(functions, values):
            temp = np.c_[temp, func(val)]
        return temp.T

    def normal_test(self):
        is_normal = {}
        for i in range(0, self.shape[1]):
            if self.var_names[i] not in ['player_id', 'team_id']:
                norm = normalize(self.player_loaded.T[i].reshape(1, -1), axis=0).ravel()
                result = stats.normaltest(norm)
                if result[1] > 0.05:
                    is_normal[self.var_names[i]] = result
        return is_normal

    def pearson_corr(self):
        pearson_correlations = {}
        for i in range(0, self.shape[1]):
            if self.var_names[i] not in ['player_id', 'team_id', 'player_value']:
                corr_coeff = stats.pearsonr(self.player_value, self.player_loaded.T[i])[0]
                if abs(corr_coeff) > 0.05:
                    pearson_correlations[self.var_names[i]] = corr_coeff
                    self.regr_names.append(self.var_names[i])
        return pearson_correlations

    def part_corr(self):
        partial_correlations = partial_corr(self.player_loaded, self.var_names, ['player_id', 'team_id'])
        return partial_correlations['player_value']

    def pc_analysis(self):
        pca_result = {}
        pca = PCA(n_components=2)
        pca_result['result'] = self.pca_reduced = pca.fit_transform(self.player_stats, self.player_value)
        pca_result['ratios'] = pca.explained_variance_ratio_
        pca_result['components'] = pca.components_
        return pca_result

    @staticmethod
    def _regression(clf, x, y):
        _clf = Pipeline([('imputer', Imputer()),
                         ('linear', clf)])
        _clf.fit(x, y)
        r_value = _clf.score(x, y)
        return r_value

    @staticmethod
    def _polynomial_regression(x, y, degree):
        clf = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                        ('linear', linear_model.LinearRegression(fit_intercept=False))])
        clf.fit(x, y)
        r_value = clf.score(x, y)
        return r_value

    def linear_regression(self):
        return self._regression(linear_model.LinearRegression(), self.player_stats, self.player_value)

    def silhouette_analysis(self):
        if not self.pca_reduced:
            self.pc_analysis()
        range_n_clusters = range(2, 10)
        for n_clusters in range_n_clusters:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)
            ax1.set_xlim([-0.1, 1])
            ax1.set_ylim([0, len(self.pca_reduced) + (n_clusters + 1) * 10])
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(self.pca_reduced)
            silhouette_avg = silhouette_score(self.pca_reduced, cluster_labels)
            print("For n_clusters =", n_clusters, "the average silhouette_score is :", silhouette_avg)
            sample_silhouette_values = silhouette_samples(self.pca_reduced, cluster_labels)
            y_lower = 10
            for i in range(n_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = cm.spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                y_lower = y_upper + 10
            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax1.set_yticks([])
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(self.pca_reduced[:, 0], self.pca_reduced[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors)
            centers = clusterer.cluster_centers_
            ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200)
            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)
            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")
            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')
            #plt.show()

    def pca_k_means(self):
        if not self.pca_reduced:
            self.pc_analysis()
        kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
        kmeans.fit(self.pca_reduced, self.player_value)
        h = .02
        x_min, x_max = self.pca_reduced[:, 0].min() - 1, self.pca_reduced[:, 0].max() + 1
        y_min, y_max = self.pca_reduced[:, 1].min() - 1, self.pca_reduced[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired, aspect='auto', origin='lower')
        plt.plot(self.pca_reduced[:, 0], self.pca_reduced[:, 1], 'k.', markersize=2)
        centroids = kmeans.cluster_centers_
        labels = self.pca_labels = kmeans.labels_
        intertia = kmeans.inertia_
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
        plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                  'Centroids are marked with white cross')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        return {'plt': plt, 'centroids': centroids, 'labels': labels, 'inertia': intertia}

    @staticmethod
    def _plot_ward_tree(x_red, x, y, labels, title=None):
        x_min, x_max = np.min(x_red, axis=0), np.max(x_red, axis=0)
        x_red = (x_red - x_min) / (x_max - x_min)

        plt.figure(figsize=(6, 4))
        for i in range(x_red.shape[0]):
            plt.text(x_red[i, 0], x_red[i, 1], str(y), color=plt.cm.spectral(labels[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        if title is not None:
            plt.title(title, size=17)
        plt.axis('off')
        plt.tight_layout()

    def pca_ward_tree(self):
        if not self.pca_reduced:
            self.pc_analysis()
        reduced_red = manifold.SpectralEmbedding(n_components=2).fit_transform(self.pca_reduced)
        clustering = AgglomerativeClustering(linkage='ward', n_clusters=3)
        clustering.fit(self.pca_reduced)
        self._plot_ward_tree(reduced_red, self.pca_reduced, self.player_value, clustering.labels_)
        return plt

    def cluster_regression(self):
        """
        Linear regression in each cluster
        :return: dict of r_values for each cluster
        """
        if not self.pca_labels:
            k_means = self.pca_k_means()
            pp.pprint(k_means)
            # k_means['plt'].show()
        clusters_pca = {}
        clusters_def = {}
        values = {}
        regr = {}
        for index, label in enumerate(self.pca_labels):
            points = np.array([self.pca_reduced[index, 0], self.pca_reduced[index, 1]])
            points_def = np.array([])
            for var in range(0, self.player_stats.shape[1]):
                points_def = np.append(points_def, self.player_stats[index, var])
            try:
                clusters_pca[label] = np.append(clusters_pca[label], [points], axis=0)
                clusters_def[label] = np.append(clusters_def[label], [points_def], axis=0)
                values[label] = np.append(values[label], self.player_value[index])
            except Exception as e:
                clusters_pca[label] = np.array([points],)
                clusters_def[label] = np.array([points_def],)
                values[label] = np.array(self.player_value[index],)


        for data in clusters_def.items():
            top = [0, 0, 0, 0, 0]
            i = 0
            for f in itertools.product(self.transforms, repeat=len(data[1])):
                i += 1
                #regr_x = self._regr_func_apply(f, data[1])
                r_value = self._regression(linear_model.LinearRegression(), data[1], values[data[0]])
                if r_value > top[4]:
                    top[4] = r_value
                    top.sort(reverse=True)
                print(i)
            print(top)
            exit()

            """

            regr[data[0]] = {}
            regr[data[0]]['x'] = self._regression(linear_model.LinearRegression(), data[1], values[data[0]])
            regr[data[0]]['ln(x)'] = self._regression(linear_model.LinearRegression(),
                                                      np.log(data[1]+0.01), values[data[0]])
            regr[data[0]]['1/x'] = self._regression(linear_model.LinearRegression(), 1/(data[1]+0.01), values[data[0]])
            regr[data[0]]['1/(1+exp(-x))'] = self._regression(linear_model.LinearRegression(),
                                                              1/(np.exp(-data[1]+0.01)), values[data[0]])
            regr[data[0]]['a^(b^x)'] = {}
            for a in range(1, 5):
                for b in range(1, 5):
                    regr[data[0]]['a^(b^x)']['a = ' + str(a) + ', b = ' + str(b)] =\
                        self._regression(linear_model.LinearRegression(), a**(b**data[1]), values[data[0]])

            for clf in self.clf:
                clf_name = str(clf).split('(')[0]
                regr[data[0]][clf_name] = self._regression(clf, data[1], values[data[0]])
                self.reset_clf()
            for degree in range(2, 6):
                regr[data[0]]['polynomial_' + str(degree)] = self._polynomial_regression(data[1],
                                                                                         values[data[0]], degree)
            """
        return regr

    def main(self):
        pp.pprint(self.cluster_regression())


analysis = PlayerAnalysis().main()
#analysis.silhouette_analysis()
#print(analysis.cluster_regression())
#print(analysis.linear_regression())
#print(analysis)