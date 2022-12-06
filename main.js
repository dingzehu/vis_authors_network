/*
Main Nodejs entry point
*/

const bodyparser = require('body-parser')
const express = require('express');
const path = require('path');
const api = require('./api/api.js') //Own api module
var app = module.exports = express();


/*--View Engine Setup--*/

// Register ejs as .html. If we did
// not call this, we would need to
// name our views foo.ejs instead
// of foo.html. The __express method
// is simply a function that engines
// use to hook into the Express view
// system by default, so if we want
// to change "foo.ejs" to "foo.html"
// we simply pass _any_ function, in this
// case `ejs.__express`.
app.engine('.html', require('ejs').__express);

// Path to our public directory
app.use(express.static(path.join(__dirname, 'public')));

// Without this you would need to
// supply the extension to res.render()
// ex: res.render('users.html').
app.set('view engine', 'html');


/*--Route our requests--*/
//From here on we handle all requests our server
//receives, they are handled async automaticity 

// body-parser libary vor easier parsing of var between
// nodejs and javascript i.e. middleware
var urlencodedParser = bodyparser.urlencoded({ extended: false })

//READ Request Handlers for our graph jsons
app.get("/api/", api.test_python);

// Link to about.html
app.get('/about', urlencodedParser, function(req, res){
  res.render('about', {
    title:"Collaborator network - About"
  })
})

//Main view
app.get('/', urlencodedParser, function(req, res){
  if (req.query.author_id === undefined){
    res.render('index', {
      title: "Collaborator network",
    });
  }
  else{
    if (req.query.embedding === undefined){
      embedding = "default";
    }
    else{
      embedding = req.query.embedding;
    }
    res.render('plot', {
      title: "Collaborator network - "+req.query.author_id,
      author_id: req.query.author_id,
      embedding: embedding,
    });    
  }
});


//Search request
app.post("/search", urlencodedParser, function(req,res){
  console.log(req.body);
  let author_id = req.body.search_string; //Parse search
  let embedding = req.body.embedding;

  let emb_types = [
        "default",
        "spectral_embedding_laplace",
        "pca_embedding",
        "node2vec_embedding",
        "kamada_kawai",
        "biHLouvain_embedding"
    ]
  if (emb_types.includes(embedding)){
    res.redirect('/?author_id='+author_id+"&embedding="+embedding);
  }
  else{
    res.redirect('/?author_id='+author_id)
  }
});



/*--Start --*/
if (!module.parent) {
	var port = process.env.port || 8080;
  app.listen(port);
  console.log(`ready on http://localhost:${port}`);
}



