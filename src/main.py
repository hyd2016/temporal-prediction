# coding=utf-8
"""
采用时间度量的半监督链接预测方法
"""

import argparse

import networkx as nx
import pandas as pd
from networkx import Graph
from typing import List

import tmlp


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph(slice_count):
    """
    Reads the input network in networkx.
    """
    col_index = ["x", "y", "time"]
    result = pd.read_table('email-Eu-core-temporal-Dept1.txt', sep=' ', header=None, names=col_index)
    max_time = result['time'].max()
    slice_num = max_time / slice_count + 1
    G_test = nx.Graph()

    test_data = result[(result['time'] >= (slice_count - 1) * slice_num) & (
                result['time'] < slice_count * slice_num)].iloc[:, 0:2]
    # 测试集
    edge_tuples = [tuple(xi) for xi in test_data.values]
    G_test.add_nodes_from(result['x'].tolist())
    G_test.add_nodes_from(result['y'].tolist())
    G_test.add_edges_from(edge_tuples)

    # 训练集
    edge_tuples = [tuple(xi) for xi in result.values]
    G = nx.Graph()
    G.add_nodes_from(result['x'].tolist())
    G.add_nodes_from(result['y'].tolist())
    G.add_weighted_edges_from(edge_tuples)
    for edge in G.edges():
        if G[edge[0]][edge[1]]['weight'] > 1:
            G[edge[0]][edge[1]]['weight'] = 1
    return result, G, G_test


def main(args):
    """
    Pipeline for representational learning for all nodes in a graph.
    """
    resent_time = 3456000
    train_data, nx_G, G_test = read_graph(20)
    G = tmlp.Graph(nx_G, train_data, G_test, args.directed, resent_time)
    G.predict()


if __name__ == "__main__":
    args = parse_args()
    main(args)
