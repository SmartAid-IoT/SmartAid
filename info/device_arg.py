device_arg = {
    'TV': {
        'off': False,
        'on': False,
        'mute': False,
        'unmute': False,
        'setMute': ['muted', 'unmuted'],
        'volumeDown': False,
        'setVolume': range(0, 101),
        'volumeUp': False,
        'setTvChannel': [0, 505, 506, 508, 516, 517, 518, 520, 524, 525, 526, 530, 534, 580, 581, 642, 643, 644, 645, 648, 651, 655, 675, 700, 731, 751, 752, 801, 831, 832, 833, 834, 835, 839],
        'setInputSource': ['digitalTv', 'HDMI2'],
        'channelDown': False,
        'channelUp': False
    },
    'bulb': {
        'on': False,
        'off': False,
        'setHue': range(0, 101),
        'setSaturation': range(0, 101),
        'setColorTemperature': range(2200, 8901, 100),
        'setLevel': range(0, 101)
    },
    'airPurifier': {
        'setAutomaticExecutionMode': ['airpurify', 'alarm'],
        'periodicSensingOn': False,
        'doNotDisturbOn': False,
        'doNotDisturbOff': False,
        'periodicSensingOff': False,
        'on': False,
        'off': False,
        'setPeriodicSensingInterval': range(600, 3001),
        'setFanMode': ['smart', 'max', 'windfree', 'sleep']
    },
    'robot': {
        'setCleaningMode': ['auto', 'stop'],
        'setRobotCleanerTurboMode': ['silence', 'on', 'off'],
        'setVolume': range(0, 101, 20),
        'pause': False,
        'returnToHome': False,
        'doNotDisturbOff': False,
        'doNotDisturbOn': False,
        'mute': False,
        'volumeUp': False,
        'setRobotCleanerMovement': ['cleaning', 'homing', 'pause'],
        'unmute': False,
        'setMute': ['muted', 'unmuted'],
        'volumeDown': False
    }
}