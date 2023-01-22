# visualisation_authors

Visualisation on co-authorship on pulications

<div id="header" align="center">
  <img src="https://github.com/dingzehu/vis_authors_network/blob/master/img/ex_page.png" width="600"/>
</div>

- Size of nodes: Number of papers published (per author)
- Size of edges: Number of papers published together (pairwise)
- Color: Institution, legend disabled because it occupied nearly the whole screen (a lot of institutions)
- Details: simple histogram (publications per year)

## Techonogies used
- Front-End: CSS, html, javscript (echarts, bootstrap)
- Middleware: NodeJS (Express, PythonShell)
- Back-End: Python (numpy, pandas, lecache, ...)
- Embedding
	- Embedding is a low-dimensional representation of high-dimensional data.
	- All embedding techniques attempt to reduce the dimensions of data, but meanwhile to preserve the "key" information in the data.
	- In this use case, the attempt is to obtain coordinates which can be presented on a 2-dimension graph, via using embedding techniques through a high-dimension adjacency matrix.
## Dataflow
<div id="header" align="center">
  <img src="https://github.com/dingzehu/vis_authors_network/blob/master/img/WorkflowOverview.png" width="600"/>
</div>

- Data pre-processing - get the proper data format
- Scrape author's ID from Semantic Scholar
- Transform processed data to adjacency matrix for further embedding, to get coordinates
- Plot embedded graph and statistic data

## How to run the website

### Installation with anaconda
The installation with conda is quite easy. This should
```bash
conda env create -f environment.yml
conda activate authors

# Install the required packages
pip install -r ./api/requirements.txt
```



### Manual installation
Installing the requirements:


```bash
# Install nodejs can also be done with binary if you do not have sudo rights
sudo apt-get install nodejs
sudo apt-get install npm

# Install the required packages
pip install -r ./api/requirements.txt

npm install
```

Open the server localy:

```bash
# Run server
nodejs main.js
```
