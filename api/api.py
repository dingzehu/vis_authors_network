import sys
import argparse
import json
import embedding  # own file with embedding functions
import numpy as np

if __name__ == "__main__":
    """
    This function gets called via api.js. The return values are passed via a
    json dump. See below.
    """
    # Parse input arguments
    parser = argparse.ArgumentParser(
        description="Returns graph json for authors in stdout!"
    )
    parser.add_argument("id", metavar="S", type=str, help="An author name or id.")
    parser.add_argument(
        "-e",
        metavar="E",
        type=str,
        help="Embedding type possible are: ['spectral_embedding_laplace',?]",
        default="default",
        required=False,
    )
    args = parser.parse_args()

    # Construct return dict/json (test for now)

    ret = embedding.calcEmbedding(args.id, args.e)

    # https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NpEncoder, self).default(obj)

    # Print it to stdout for further processing with nodejs
    print(json.dumps(ret, cls=NpEncoder))
