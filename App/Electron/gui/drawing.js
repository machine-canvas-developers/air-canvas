const { spawn } = require('child_process')
const path = require( "path" )



function start_draw(){
const {PythonShell} = require('python-shell');

let pyshell = new PythonShell(`${path.join(__dirname, "../")}/engine/drawing.py`);

pyshell.send(JSON.stringify([10]))

pyshell.on('message', function(message) {
  console.log(message);
})

pyshell.end(function (err) {
  if (err){
    throw err;
  };
  console.log('finished');
});
}

// function start_draw() {
    
//     const childProcess = spawn('python', [`${path.join(__dirname, "../../")}/engine/drawing.py`])


//     childProcess.stdout.on('data', (data) => {
//         console.log(`stdout: ${data}`);
//     });
    
//     childProcess.stderr.on('data', (data) =>{
//         console.error(`stderr: ${data}`);
//     });
    
//     childProcess.on('close', (code) =>{
//         console.log(`Child process exited with code ${code}`);
//     });

      
// };

// function start_gesture() {

//     const childProcess = spawn('python', [`${path.join(__dirname, "../../")}/engine/prediction.py`])


//     childProcess.stdout.on('data', (data) => {
//         console.log(`stdout: ${data}`);
//     });
    
//     childProcess.stderr.on('data', (data) =>{
//         console.error(`stderr: ${data}`);
//     });
    
//     childProcess.on('close', (code) =>{
//         console.log(`Child process exited with code ${code}`);
//     });
      
// };

// start_draw()

// alert("Something is working")

// let {PythonShell} = require('python-shell')
// // var path = require("path")

// console.log("Working??")
// function start_draw() {

// //   var city = document.getElementById("city").value
  
//   var options = {
//     scriptPath : `{path.join(__dirname, "../../")}/engine/prediction.py`,
//     //args : [city]
//   }

//   let pyshell = new PythonShell('drawing.py', options);


//   pyshell.on('message', function(message) {
//     swal(message);
//   })
// //   document.getElementById("city").value = "";
  
// }

// start_draw()


