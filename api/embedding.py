# -*- coding: utf-8 -*-
# @Author: Sebastian B. Mohr
# @Date:   2021-07-15 11:53:25
# @Last Modified by:   Sebastian Mohr
# @Last Modified time: 2021-07-22 13:44:30

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from node2vec.model import Node2Vec
from sknetwork.embedding import HLouvainEmbedding, BiHLouvainEmbedding
from filecache import filecache
import networkx as nx
import os

from utils import getAdjacencyMatrixAndNodes


# Change path to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


@filecache
def get_Afull():
    # Load full adjacency matrix with caching because file load time are annoying
    return pd.read_csv("../kaggle_dataset/first_try/connection_matrix.csv", index_col=0)


@filecache
def get_Color():
    return pd.read_csv("../kaggle_dataset/clean_data/institutions_color.csv")


@filecache
def get_Size():
    return pd.read_csv("../kaggle_dataset/clean_data/authors_size.csv")


@filecache
def get_Id2Name():
    return pd.read_csv(
        "../kaggle_dataset/clean_data/semanticID_authors.csv", dtype=str
    ).dropna()


def getAdjacencyMatrixAndNodes(semantic_id, n=2):
    """

    Parameters
    ----------
    semantic_id : str
        Semantic id of author
    n : int
        Depth of adjacency matrix

    Returns
    -------
    Adjancency matrix, nodes dict
    """

    # Get author name from id
    id_2_name = get_Id2Name()
    name = id_2_name[id_2_name["authorid"] == semantic_id]["name"]
    if len(name) != 1:
        # print("ID not found")
        raise ValueError(f"ID '{semantic_id}' was not found in kaggel dataset!")
    name = name.values[0]

    # Load full adjacency kaggle dataset
    A_full = get_Afull()

    def recursion(name, authors_full=[], n_steps=0):
        authors = A_full.index[A_full[name] >= 1]
        authors_full = np.unique(list(authors) + list(authors_full) + [name])
        if n_steps == n:
            return authors_full

        for author in authors:
            return recursion(author, authors_full, n_steps + 1)

    # Extract connected authors up to depth n
    authors = recursion(name)

    # Extract smaller adjacency matrix from big full dataset
    A = A_full.loc[authors][authors]

    # Create node dictionary
    d = {}
    nodes = []

    # load dataset with colours of institutions
    color = get_Color()

    # load dataset with paper_count of each authors
    size = get_Size()

    # Construct institution (for category plotting)
    categories = []
    unique_institutions = np.unique(
        color.loc[color["name_in_database"].isin(authors), "institution"].values
    )
    for ins in unique_institutions:
        categories.append({"name": ins})

    # construct nodes
    for a, author in enumerate(authors):
        try:
            sid = id_2_name[id_2_name["name"] == author]["authorid"].values[0]
        except:
            sid = str(a)

        ins = color.loc[color["name_in_database"] == author, "institution"].values[0]

        # to number
        ins = np.where(unique_institutions == ins)

        temp = {
            "id": sid,
            "label": author,
            "size": np.sqrt(
                size.loc[size["name_in_database"] == author, "paper_count"].values[0]
            ),
            "institution": ins,
        }
        nodes.append(temp)

    # Construct edges
    edges = []
    for i, a in enumerate(np.triu(A)):  # only upper triangle to discard double edges
        for j, b in enumerate(a):
            if b >= 1:
                edge = {
                    "sourceID": nodes[j]["id"],
                    "targetID": nodes[i]["id"],
                    "size": b,
                }
                edges.append(edge)

    d["nodes"] = nodes
    d["edges"] = edges
    d["categories"] = categories
    return A, d


def calcEmbedding(semantic_id, embedding):
    """
    High level function to calculate an embedding by a given author id. Is the main
    entry point for most of the api plotting calls.

    Parameters
    ----------
    semantic_id : str,
    embedding: str,
        Type of embedding to calculate possible values are
        ["spectral_embedding_laplace","pca_embedding","node2vec_embedding","default"].
    """

    # Check for valid embedding type

    emb_types = [
        "default",
        "spectral_embedding_laplace",
        "pca_embedding",
        "node2vec_embedding",
        "kamada_kawai",
        "biHLouvain_embedding",
    ]
    if embedding not in emb_types:
        raise ValueError(
            f"Embedding type {embedding} not valid! Possible: {emb_types}!"
        )

    # Get weight/adjacency matrix by semantic author id and node dictionary
    A, authorNodesEdges = getAdjacencyMatrixAndNodes(semantic_id)
    A = np.array(A)
    # Calculate x,y coordinates for chosen embedding using
    # the weight/adjacency matrix
    if embedding == "default":
        embedding = "kamada_kawai"

    if embedding == "spectral_embedding_laplace":
        X = spectral_embedding_laplace(A)
    elif embedding == "pca_embedding":
        X = pca_embedding(A).T
    elif embedding == "node2vec_embedding":
        X = node2vec_embedding(A).T
    elif embedding == "kamada_kawai":
        X = kamada_kawai(A).T
    elif embedding == "biHLouvain_embedding":
        X = biHLouvain_embedding(A).T

    # Scale X between [0,1000]
    X = (
        (X - X.min(axis=1)[:, None])
        / (X.max(axis=1)[:, None] - X.min(axis=1)[:, None])
        * 1000
    )

    # put computed node positions into NodeEdge dictionary
    for i, node in enumerate(authorNodesEdges["nodes"]):
        node["x"] = X[0, i]
        node["y"] = X[1, i]

    return authorNodesEdges


def kamada_kawai(w):
    G = nx.Graph(w)
    return np.array(list(nx.kamada_kawai_layout(G).values()))


def spectral_embedding_laplace(w, dim=2, return_full=False):
    """
    Performs spectral embedding with Laplacian.

    Parameters
    ----------
    w : np.array, shape (NxN)
        Weight- or adjacency matrix with the encoded node to
        node connections.
    dim : int
        The dimension mapping, i.e. on which dimension should the graph
        be embedded onto.

    Returns
    -------
    x : np.array, shape (2,N)
        Coordinates of each node
    """

    # compute laplacian
    L = -w
    deg = np.sum(w, axis=1)
    np.fill_diagonal(L, deg)

    # Normalize by degree
    L = np.einsum(L, [0, 1], deg ** (-0.5), [0], deg ** (-0.5), [1], [0, 1])

    # compute eigenvectors/eigenvalues
    vals, vectors = np.linalg.eigh(L)

    # sort by size
    i = vals.argsort()[::-1]
    vals = vals[i]
    vectors = vectors[i]

    # Get the n largest ones
    X = []
    for i in range(dim):
        X.append(vectors[:, i + 1])

    if return_full:
        return vectors
    return np.array(X)


def pca_embedding(w):
    """
    Parameters
    ----------
    w : np.array, shape (NxN)
        Weight- or adjacency matrix with the encoded node to
        node connections.
    """

    # PCA stpe by step
    X_std = StandardScaler().fit_transform(w)

    mean_vec = np.mean(X_std, axis=0)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    u, s, v = np.linalg.svd(X_std.T)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    # return Nx2 matrix
    matrix_w = np.hstack(
        (eig_pairs[0][1].reshape(w.shape[0], 1), eig_pairs[1][1].reshape(w.shape[0], 1))
    )

    return matrix_w


def node2vec_embedding(w):
    """
    Node 2 vec embedding after https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf
    using the package by 
    """
    # Create a node2vec model from the edgelist
    node2vec_model = Node2Vec.from_adj_matrix(w)

    # Simulate biased random walks on the graph
    node2vec_model.simulate_walks(
        walk_length=40, n_walks=200, p=1, q=0.6, workers=4, verbose=False
    )

    # Learn node embeddings from the generated random walks
    node2vec_model.learn_embeddings(
        dimensions=124, context_size=10, epochs=50, workers=4, verbose=False
    )

    # Learn a projection from 128 dimensions to 2
    tsne = TSNE(n_components=2, perplexity=10, early_exaggeration=1)
    X_transformed = tsne.fit_transform(node2vec_model.embeddings)

    # Print the embedding corresponding to the first node
    return X_transformed


def biHLouvain_embedding(w):

    bilouvain = BiHLouvainEmbedding(2)
    embedding = bilouvain.fit_transform(w)

    return embedding


if __name__ == "__main__":

    """
    Test spectral_embedding_laplace
    """
    # three nodes with ab. connections
    semantic_id = "48920094"

    A, nodes = getAdjacencyMatrixAndNodes(semantic_id)
    A = np.array(A)
    A = np.zeros((14, 14))
    for i in range(13):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    A[13, 0] = 1
    A[0, 13] = 1

    def plot(x, y, A, title=""):
        # Remove axis and set figuresize
        plt.subplots(figsize=(10, 10))
        plt.axis("off")

        # Plot edges
        for i, b in enumerate(A):
            for j, a in enumerate(b):
                if a > 0:
                    x_s = [x[i], x[j]]
                    y_s = [y[i], y[j]]
                    plt.plot(x_s, y_s, color="tab:gray")

        # Plot points
        plt.scatter(x, y, marker="o", s=150, zorder=10, color="tab:green")
        plt.title(title)

    """
    Test and show all embeddings
    """
    import matplotlib.pyplot as plt

    X = pca_embedding(A).T
    plot(X[0], X[1], A, title="PCA")
    plt.show()

    X = spectral_embedding_laplace(A)
    plot(X[0], X[1], A, title="Spectral Laplace")
    plt.show()

    X = node2vec_embedding(A).T
    plot(X[0], X[1], A, title="Node2Vec")
    plt.show()
