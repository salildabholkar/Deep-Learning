# Game of Life

This game requires no players. 
The only input to the game is its initial configuration, which in this case is generated automatically

The cells transition using certain rules, and these rules can be easily learnt using a simple neural network. 
One can observe how the game evolves by creating patterns having certain properties.

Blinker pattern          |  Colony Simulation
:-------------------------:|:-------------------------:
![Neural Net generating a pattern](life2.gif?raw=true "Neural Net generating a pattern")  |  ![Neural Net generating the Game of life](life.gif?raw=true "Neural Net generating the Game of life")
 

The training data was created by taking random combinations of the neighbors and
checking the cell transition given the combination.<br/>
The neural net could accurately reproduce the above shown as well as all other known patterns

## Running the project

### Requirements
* python 3
* pip 3

### Clone the project
Clone the project to your local machine using:

    git@github.com:salildabholkar/Deep-Learning.git

### Install dependencies:
Navigate to the `GameOfLife` directory

Make sure `python 3` is being used by doing `python --version`

Run the following command to install required dependancies:

    pip install -r frozen-requirements.txt

### Start the game
In the `GameOfLife` directory, use the following command to see the game of life in action:

    python .