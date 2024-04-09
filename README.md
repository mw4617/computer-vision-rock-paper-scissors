# Rock Paper Scissors AI Game

##TOC
1. Description
2. Installation Instructions
3. Usage Instructions
4. File Structure
5. License Information

## 1. Description

This project is a Rock, Paper, Scissors game where a human user plays against an AI. There is two versions to the game. One simplfied version ( manual_rps.py ) where user enters the selected hand gesture throught the command line when prompted and more advanced (camera_rps.py) version where user literally plays rock paper scissors game, making appriopriate hand gestures against the pc. This more advanced version of the game uses a camera to capture the user's hand gestures, classifies them as rock, paper, or scissors, and generates a computer response based on the gesture. It incorporates machine learning models to improve gesture recognition accuracy, including separate models for different lighting conditions. This project aims to demonstrate the application of computer vision and machine learning in creating interactive applications. Through this project, I learned about model training (teachable machine learning), image preprocessing, and integrating Python scripts with real-time video input.

## 2. Installation Instructions
To set up the Rock Paper Scissors AI Game, follow these steps:
- Ensure Python 3.x is installed on your system.
- Install necessary libraries using pip:
  ```
  pip install opencv-python keras numpy click
  ```
- Clone the repository or download the game files to your local machine.

## 3. Usage Instructions
To play the game, run the script from the command line:
```
python camera_rps.py --rounds 5 --lighting daylight
```
Options:
- `--rounds` specifies the number of game rounds (default is 3).
- `--lighting` sets the lighting condition (`daylight`, `artificial`, `unspecified`). If unspecified AI model will try classify itself the lighting conditions, although beware this is not recommended and give flawed results.

Once the game has started simply show the right hand shape to camera and press q on the keyboard. Follow the onscreen instructions.

## 4. File Structure
The project structure is as follows:
- `manual_rps.py`: The main game script simplified command prompt input (manual) version.
- `camera_rps.py`: The main game script AI version.
- `keras_model.h5`: Pre-trained model for gesture recognition no 1.
- `keras_model_2.h5`: Pre-trained model for gesture recognition no 2.
- `keras_model_3.h5`: Pre-trained model for gesture recognition no 3.
- `keras_model_4.h5`: Pre-trained model for gesture recognition no 4.
- `keras_model_5.h5`: Pre-trained model for gesture recognition no 5.
- `keras_model_lighting_1.h5`: Model for lighting condition classification no 1.
- `keras_model_lighting_2.h5`: Model for lighting condition classification no 2.
- `labels.txt`: labels and their numeric representation for the hand gesture classification model
- `labels_lighting.txt`: labels and their numeric representation for the lighting model
- `README.md `: this manual

## 5. License Information
Free to download and play with, but not to edit and republish without my explicit consent.
