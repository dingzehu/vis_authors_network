
//Utils
var getJSON = function(url, callback) {
  var xhr = new XMLHttpRequest();
  xhr.open('GET', url, true);
  xhr.setRequestHeader("Content-Type", "application/json");
  xhr.responseType = 'json';
  xhr.onload = function() {
    var status = xhr.status;
    if (status === 200) {
      callback(null, xhr.response);
    } else {
      callback(status, xhr.response);
    }
  };
  xhr.send();
};

Array.prototype.remove = function() {
  var what, a = arguments, L = a.length, ax;
  while (L && this.length) {
    what = a[--L];
    while ((ax = this.indexOf(what)) !== -1) {
      this.splice(ax, 1);
    }
  }
  return this;
};

Array.prototype.max = function() {
  return Math.max.apply(null, this);
};

Array.prototype.min = function() {
  return Math.min.apply(null, this);
};

function onlyUnique(value, index, self) {
  return self.indexOf(value) === index;
}

function addEvent(element, eventName, fn) {
  if (element.addEventListener)
    element.addEventListener(eventName, fn, false);
  else if (element.attachEvent)
    element.attachEvent('on' + eventName, fn);
}


function hexToRgb(hex) {
  var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
}


function tint_rgb(dic, factor){
  dic["r"] = dic["r"] + (255 - dic["r"]) * factor;
  dic["g"] = dic["g"] + (255 - dic["g"]) * factor;
  dic["b"] = dic["b"] + (255 - dic["b"]) * factor;
  return dic
}

var pubChart;

var plotPublication = function(div, counts){
  pubChart = echarts.init(div, null, {
    renderer: 'svg'
  });

  pubChart.setOption(option = {
    title: {
        text: 'Number of publications',
        textStyle: {
          color: '#272832',
          fontFamily: "Iter",
          fontWeight: "600",
          fontSize: "1rem",
        },
        left: 'center',
        padding:0,
        top:'25',
    },
    textStyle: {
      color: '#272832',
      fontFamily: "Iter",
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: {           
        type: 'shadow'        //'line' | 'shadow'
      },
      backgroundColor : "#DFE6E660",
      borderColor: "#333",
      textStyle: {
          color: '#272832',
          fontFamily: "Iter",
          fontWeight: "600",
          fontSize: "normal",
        },
    },
    xAxis: {
      type: 'category',
      data: Object.keys(counts),
      axisTick: {
        alignWithLabel: true
      },
    },
    yAxis: {
      axisLine: {
        show: false
      },
      axisTick: {
        show: false
      },
      splitLine:{
        lineStyle: {
          color: '#272832',
          opacity: 0,
        }
      },
      type: 'value',
      minInterval: 1,
    },
    series: [{
      type: 'bar',
      data: Object.values(counts),
      itemStyle: {
        color: "#4D1787",
      }
    }],
  }, true);
  addEvent(window, 'resize', function(){ pubChart.resize();});
}

// Setting author details div

var setDetails = function(semanticId, name, aliases, papers, numInfluentialCitations){

  let childs = [];
  /* -- Title with name -- */
  title = document.createElement("h4");
  title.classList.add("card-title");
  title.textContent = name;
  childs.push(title);

  /* -- Subtitle aliases -- */
  aliases.remove(name);

  if (aliases.length > 0) {
    subtitle = document.createElement("p");
    subtitle.classList.add("text-muted");
    subtitle.classList.add("card-subtitle")
    subtitle.textContent = "Also known as ";
    for (let i = 0; i < aliases.length; i++){
      if (i == 0){
        subtitle.textContent += aliases[i];
      }
      else if (i != aliases.length - 1) {
        subtitle.textContent += ", "+aliases[i];
      }
      else{
        subtitle.textContent += " or "+aliases[i];
      }
    }
    subtitle.textContent += ".";
    childs.push(subtitle);
  }

  childs.push(document.createElement("hr"));


  /* -- Link to semantic scholar -- */
  link_collabNet = document.createElement("a");
  link_collabNet.href = "?author_id="+semanticId;

  links_div = document.createElement("div");
  links_div.classList.add("badge");
  links_div.classList.add("social-link");

  link_collabNet_icon = document.createElement("span");
  link_collabNet_icon.classList.add("ai");
  link_collabNet_icon.classList.add("ai-2x");
  link_collabNet_icon.classList.add("iconSelf");

  links_div.append(link_collabNet_icon);
  link_collabNet.append(links_div);
  childs.push(link_collabNet);

  /* -- Link to semantic scholar -- */
  link_semantic = document.createElement("a");
  link_semantic.target = "_blank"; //New tab
  link_semantic.rel = "noopener";
  link_semantic.href = "https://www.semanticscholar.org/author/"+semanticId;
  
  links_div = document.createElement("div");
  links_div.classList.add("badge");
  links_div.classList.add("social-link");

  link_semantic_icon = document.createElement("i");
  link_semantic_icon.classList.add("ai");
  link_semantic_icon.classList.add("ai-semantic-scholar");
  link_semantic_icon.classList.add("ai-2x");

  links_div.append(link_semantic_icon);
  link_semantic.append(links_div);
  childs.push(link_semantic);





  childs.push(document.createElement("hr"));
  /* -- Num publications per year -- */
  years = []; //Get all years
  
  for (let paper of papers){
    years.push(paper.year);
  }
  // Count the number of publications each year
  let counts = {};
  years.forEach(function (x) { counts[x] = (counts[x] || 0) + 1; });
  delete counts['null']

  for (let i = Object.keys(counts).min(); i<Object.keys(counts).max();i++){
    if (counts[i] === undefined){
      counts[i] = 0
    }
  }

  console.log(counts);
  publications_over_time = document.createElement("div");
  publications_over_time.classList.add("card-text");
  publications_over_time.style.height='300px';
  //Plot see below
  childs.push(publications_over_time);
  


  /*-Add to details div-*/
  details = document.getElementById("details")
  while (details.firstChild){ //Remove all old ones
    details.removeChild(details.firstChild);
  }
  for (let child of childs){
    details.append(child);
  }

  //For some reason the plot have to be created after adding to the 
  // body
  plotPublication(publications_over_time, counts);
}

var getAuthorDetails = function(id){
  /* Gets author details by semantic scholar id */

  url = "https://api.semanticscholar.org/v1/author/" + id

  getJSON(url, function(err, data) {
    if (err !== null) {
      console.log("Could not get author details! Something wrong with the api?" + err);

      /* -- Create goodish looking error page -- */
      let childs = []
      title = document.createElement("h4");
      title.classList.add("card-title");
      title.textContent = id;
      childs.push(title);

      childs.push(document.createElement("hr"));

      body = document.createElement("div");
      body.classList.add("card-text");
      body.textContent = "Could not load the author details! " + err; 
      childs.push(body);
      
      while (details.firstChild){ //Remove all old ones
        details.removeChild(details.firstChild);
      }
      for (let child of childs){
        details.append(child);
      }
    } else {
      setDetails(id, data.name, data.aliases, data.papers, data.influentialCitationCount)
    }
  })
}