# smartAID
This project is an implementation of smartAID.

## Prerequisties
Our implementation is based on Pytorch 3.8 and Pytorch 1.12.1.
Please see the full list of packages required to our codes in requirements.txt

## Datasets
The dataset is collected from SmartThings platform, a famous IoT platform with 62 million users.
This is the first dataset for studying action planning for smart home.
### Logs
We provide device control logs of 4 devices in the folder 'log': bulb, TV, robot cleaner, and air purifier.
The log of each device is a data frame of length N, where N is the number of instances. Each instance has 'prev', 'cmd', and 'next' columns. 'prev' column is the state of the device before the execution of the command.
'cmd' column has the name of the command that is executed and the argument that is used.
'next' column is the state of the device after the execution of the command.
### Arguments
<code/>arguments.py</code> is a dictionary of possible arguments for each command of devices.
### Capability Values
<code/>cap_value.py</code> is a dictionary of possible values for each capability of devices.

## Running the code
You can train the models for smartAID by running <code/>python train.py</code> in <code/>src</code> folder with arguments <code/>--device</code> and <code/>--save</code>. Set <code/>--device</code> as one among 'bulb', 'TV', 'robotCleaner', and 'airPurifier'. If you do not set <code/>--device</code>, the default device is 'bulb. Set <code/>--save</code> argument to 'False' to not save the results. The default setting is 'True'.

After training the models, you can do action planning by running <code/>python plan.py</code> with arguments <code/>--device</code>, <code/>--num</code>, and <code/>--save</code>.
Set <code/>--num</code> as the number of random states to do action planning on. If you do not set <code/>--num</code>, the default number is 1000.

We provide <code/>demo.sh</code>, which trains the model and creates 100 plans on device 'bulb'.
