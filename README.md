# visualisation_authors

Project for Visualisation Exam




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
