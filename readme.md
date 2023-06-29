# SmartAID
This project is an implementation of SmartAID.

## Prerequisties
Our implementation is based on Pytorch 3.8 and Pytorch 1.12.1.
Please see the full list of packages required to our codes in requirements.txt

## Datasets
The dataset is collected from SmartThings platform, a famous IoT platform with 62 million users.
This is the first dataset for studying action planning for smart home.
We provide device control logs of 4 devices: bulb, TV, robot cleaner, and air purifier.

## Running the code
You can train the models for smartAID by <code/>python train_smartAID.py</code> with arguments <code/>--device</code> and <code/>--save</code>. Set <code/>--device</code> as one among 'bulb', 'TV', 'robot', and 'airPurifier'. Set <code/>--save</code> argument as <code/>True</code> to save the trained model and <code/>False</code> to not.

After training the models, you can do action planning by running <code/>python plan.py</code> with arguments <code/>--device</code>, <code/>--distance</code>, and <code/>--save</code>.
Set <code/>--distance</code> as a number to set the distance between start states and target states. If you do not set distance, it automatically plans from distance 1 to 5.

You can also try action planning with other methods such as 'Astar' and 'BFS' by setting the argument <code/>--name</code>.
