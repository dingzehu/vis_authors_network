//Import PythonShell module.
let {PythonShell} = require('python-shell')

exports.test_python = function test_python(req,res){
    // Use child_process.spawn method from 
    // child_process module and assign it
    // to variable spawn
    var spawn = require("child_process").spawn;
      
    // Parameters passed in spawn -
    // 1. type_of_script
    // 2. list containing Path of the script
    //    and arguments for the script 
      
    // E.g : ?author='Sebastian B Mohr'
    // so, first name = Mike and last name = Will
    console.log(req.query);

    var embedding = "default";
    if (req.query.embedding !== undefined) {
        embedding = req.query.embedding
    }

    let options = {
      mode: 'text',
      pythonOptions: ['-u'], // get print results in real-time
      scriptPath: './api/',
      args: [req.query.author, "-e"+embedding],
    };

    PythonShell.run('api.py', options, function (err, result) {
      if (err) throw err;
      console.log('result: ', result.toString())
      // results is an array consisting of messages collected during execution
      res.json(JSON.parse(result.toString()));
    });
}
