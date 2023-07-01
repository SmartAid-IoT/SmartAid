# SmartAID
This project is an implementation of SmartAID.

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
You can train the models for smartAID by running <code/>python train.py</code> in <code/>src</code> folder with arguments <code/>--device</code> and <code/>--save</code>. Set <code/>--device</code> as one among 'bulb', 'TV', 'robotCleaner', and 'airPurifier'. If you do not set <code/>--device</code>, the default device is 'bulb'. Set <code/>--save</code> argument to 'False' to not save the results. The default setting is 'True'.

After training the models, you can do action planning by running <code/>python plan.py</code> with arguments <code/>--device</code>, <code/>--num</code>, and <code/>--save</code>.
Set <code/>--num</code> as the number of random states to do action planning on. If you do not set <code/>--num</code>, the default number is 1000.

We provide <code/>demo.sh</code>, which trains the model and creates 100 plans on the device 'bulb'. The result of the <code/>demo.sh</code> would be like below:

```
===== BULB =====

AVAILABLE COMMANDS:
1. on
2. setLevel
3. off
4. setHue>1
5. setSaturation>1
6. setColorTemperature>1

--- TRAIN: on ---
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:01<00:00, 25.96it/s]
Accuracy: 1.0

--- TRAIN: setLevel ---
 43%|██████████████████████████████████████████▉                                                        | 13/30 [00:10<00:14,  1.20it/s]
Accuracy: 1.0

--- TRAIN: off ---
 62%|█████████████████████████████████████████████████████████████▉                                     | 25/40 [00:07<00:04,  3.50it/s]
Accuracy: 1.0

--- TRAIN: setHue>1 ---
 93%|████████████████████████████████████████████████████████████████████████████████████████████▍      | 28/30 [00:10<00:00,  2.59it/s]
Accuracy: 1.0
 50%|█████████████████████████████████████████████████▌                                                 | 15/30 [00:05<00:05,  2.53it/s]
Accuracy: 1.0

--- TRAIN: setSaturation>1 ---
 37%|████████████████████████████████████▎                                                              | 11/30 [00:04<00:07,  2.43it/s]
Accuracy: 0.983451536643026
 43%|██████████████████████████████████████████▉                                                        | 13/30 [00:05<00:06,  2.45it/s]
Accuracy: 1.0

--- TRAIN: setColorTemperature>1 ---
 40%|███████████████████████████████████████▌                                                           | 12/30 [00:08<00:12,  1.42it/s]
Accuracy: 1.0

Average Regression Score: 1.0
Average Classification Score: 1.0

=== RESULT ===
                        types                                       target  ...    test_len                                     model
setLevel                  [r]                               (switchLevel,)  ...       [870]                      [LinearRegression()]
setHue>1               [r, r]  (colorControl_hue, colorControl_saturation)  ...  [417, 417]  [LinearRegression(), LinearRegression()]
setColorTemperature>1     [r]                          (colorTemperature,)  ...       [730]                      [LinearRegression()]
on                        [c]                                    (switch,)  ...        [51]                    [LogisticRegression()]
off                       [c]                                    (switch,)  ...       [355]                    [LogisticRegression()]
setSaturation>1        [r, r]  (colorControl_hue, colorControl_saturation)  ...  [423, 423]  [LinearRegression(), LinearRegression()]

[6 rows x 7 columns]


===== BULB =====

100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 53.36it/s]
           length        time  memory_num  visited_num
count  100.000000  100.000000  100.000000   100.000000
mean     3.770000    0.017304  111.160000   193.380000
std      0.722719    0.006501   46.230367    62.382136
min      2.000000    0.003087   32.000000    48.000000
25%      3.000000    0.012702   72.750000   152.500000
50%      4.000000    0.017671  101.000000   200.500000
75%      4.000000    0.022289  154.250000   231.750000
max      5.000000    0.029956  201.000000   311.000000
```
