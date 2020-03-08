# coding=utf-8
import numpy as np
import networkx as nx
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, nx_G, train_data, test_G, is_directed, resent_time):
        self.G = nx_G
        self.is_directed = is_directed
        self.train_data = train_data
        self.test_G = test_G
        n = len(nx_G)
        self.edge_info = []
        self.edge_category = np.zeros(n ** 2, dtype=np.int)
        self.edge_feature = np.zeros((n ** 2, 7))
        self.resent_time = resent_time

    def compute_feature_metrics(self):
        """
        计算特征拓扑
        edge_info:用于索引是哪条边，{index:(x,y),...}
        edge_category:用于存放边的类别，矩阵，大小为（n**2，1）
        edge_feature:边特征矩阵，大小为(n*n, 7)
        :return: None
        """
        # n = len(self.G.nodes())
        # edge_info = []
        # edge_category = np.zeros(n ** 2, dtype=np.int)
        # edge_feature = np.zeros((n ** 2, 7))
        index = 0
        for x in self.G.nodes():
            for y in self.G.nodes():
                self.edge_info.append((x, y))
                if (x, y) in self.G.edges():
                    self.edge_category[index] = 1
                common_neighbors = len(list(nx.common_neighbors(self.G, x, y)))
                self.edge_feature[index][0] = common_neighbors
                try:
                    self.edge_feature[index][1] = nx.shortest_path_length(self.G, x, y)
                except:
                    self.edge_feature[index][1] = -1
                self.edge_feature[index][2] = common_neighbors / len(
                    set(nx.neighbors(self.G, x)) | set(nx.neighbors(self.G, y)))
                index = index + 1
        # self.edge_info = edge_info
        # self.edge_category = edge_category
        # self.edge_feature = edge_feature
        return

    def compute_time_feature(self):
        """
        计算时间特质，包括最近访问时间，回报率，活跃度，平均活跃度
        :return:
        """
        max_time = self.train_data['time'].max()
        self.train_data['time'] = self.train_data['time'] - max_time
        self.train_data.sort_values('time', ascending=False)
        weight_edges = [tuple(edge) for edge in self.train_data.values]
        # 总的数据集构成图
        G_t = nx.Graph()
        G_t.add_nodes_from(self.G.nodes())
        G_t.add_weighted_edges_from(weight_edges)
        resent_edge = self.train_data[self.train_data['time'] < self.resent_time]
        # 最近访问时间生成的图
        G_r = nx.Graph()
        G_r.add_nodes_from(self.G.nodes())
        resent_edges_tuple = [tuple(edge) for edge in resent_edge.values]
        G_t.add_weighted_edges_from(resent_edges_tuple)
        index = 0
        for x in G_t.nodes():
            for y in G_t.nodes():
                if (x, y) in G_t:
                    self.edge_feature[index][3] = G_t[x][y]['weight']
                ret_x = (len(nx.neighbors(G_t, x)) - len(nx.neighbors(G_r, x)))/len(nx.neighbors(G_t, x))
                ret_y = (len(nx.neighbors(G_t, y)) - len(nx.neighbors(G_r, y)))/len(nx.neighbors(G_t, y))
                self.edge_feature[index][4] = ret_x + ret_y
                active_x = len(nx.neighbors(G_r, x)) / self.resent_time
                active_y = len(nx.neighbors(G_r, y)) / self.resent_time
                self.edge_feature[index][5] = active_x + active_y
                self.edge_feature[index][6] = active_x + active_y
                index = index + 1

    def predict(self):
        self.compute_feature_metrics()
        self.compute_time_feature()
        x_train, x_test, y_train, y_test = train_test_split(self.edge_feature, self.edge_category, random_state=1,
                                                            train_size=0.5)
        clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
        clf.fit(x_train, y_train.ravel())
        y_pred = clf.predict(x_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, marker='o')
        plt.show()
        auc_score = roc_auc_score(y_test, y_pred)
        print auc_score
        return