# -*- coding: utf-8 -*-
# @Author: Sebastian B. Mohr
# @Date:   2021-07-19 19:48:36
# @Last Modified by:   Sebastian Mohr
# @Last Modified time: 2021-07-19 20:00:19
import numpy as np


def getAuthorIdFromSemanticScholar(str):
    import requests

    # Maybe we can somehow design the payload to make the request
    # smaller. I reverse enginiered the following:
    payload = {
        "queryString": '"Sebastian B. Mohr"',
        "page": 1,
        "pageSize": 1,
        "sort": "relevance",
        "authors": [],
        "coAuthors": [],
        "venues": [],
    }
    url = "https://www.semanticscholar.org/api/1/search"

    response = requests.post(url, json=payload)
    # More official calls https://api.semanticscholar.org/
    return response.json()["matchedAuthors"]


def getAdjacencyMatrixAndNodes(semantic_id):
    """
    Should return adjecency matrix and nodes/edges dict. For now
    only returns test values.
    """

    default_nodes = {
        "nodes": [
            {
                "color": "#4f19c7",
                "label": "Foo Bar",
                "x": 0,
                "y": 0,
                "id": "01",
                "size": 20,
            },
            {
                "color": "#ff19c7",
                "label": "Bar",
                "x": 2,
                "y": 3,
                "id": "02",
                "size": 10,
            },
            {
                "color": "#ff19c7",
                "label": "Foo",
                "x": -2,
                "y": -3,
                "id": "03",
                "size": 10,
            },
        ],
        "edges": [
            {"sourceID": "01", "targetID": "02", "size": 1},
            {"sourceID": "01", "targetID": "03", "size": 1},
        ],
    }

    A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])

    return A, default_nodes
