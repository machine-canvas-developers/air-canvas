# Project - Air Canvas

* Air canvas is a free tool using which you can freely draw on air!
* It recognizes your hand instantly after which you will be able to draw anything on screen.
* It also comes with hand gesture recognition thru which you can give command via hand gestures.

### Built With
* [python](https://www.python.org/downloads/) - Cross platform language
* [OpenCV](https://pypi.org/project/opencv-python/) - To work with webcam
* [MediaPipe](https://google.github.io/mediapipe/solutions/hands) - Detecting contours 
* [Tensorflow](https://www.tensorflow.org/) - Training/testing model
* [Electron](https://www.electronjs.org/) - Desktop GUI

### Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required libraries.

```bash
pip install opencv-contrib-python
pip install mediapipe
pip install tensorflow
pip install numpy
pip install keras
pip install sklearn
```

### Setup
To run this project, install it locally using npm
```
$ npm install
$ npm start
```


### Usage
It is very easy to use:
* Draw tab - Join your index finger and thumb to start drawing. And using index finger you can select the options available in the GUI.

* Gesture tab - Make a gesture in your hand, app will detect it and work accordingly. 

> Currently, there are only 2 gestures available i.e. Fist (Play current song), Palm (Pause current song). We have plans to add more gestures.

### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

