config = {
    'bulb': {
        ("switch", ): (["on", "off"], ),
        ("colorTemperature", ): (range(2200, 8901, 100), ),
        ("switchLevel", ): (range(0, 101), ),
        ("colorControl_hue", ): (range(0, 101), ),
        ("colorControl_saturation", ): (range(0, 101), )
    },
    'TV': {
        ("switch", ): (["on", "off"], ),
        ("audioVolume", ): (range(0, 101), ),
        ("audioMute", ): (["unmuted", "muted"], ),
        ("tvChannel", ): ([0, 505, 506, 508, 516, 517, 518, 520, 524, 525, 526, 530, 534, 580, 581, 642, 643, 644, 645, 648, 651, 655, 675, 700, 731, 751, 752, 801, 831, 832, 833, 834, 835, 839], ),
        ("mediaInputSource", ): (["digitalTv", "HDMI2"], )
    },
    'robot': {
        ("robotCleanerTurboMode", ): (["silence", "on", "off"], ),
        ("audioVolume", ): (range(0, 101, 20), ),
        ("audioMute", ): (["unmuted", "muted"], ),
        ("custom.doNotDisturb", ): (["on", "off"], ),
        ("samsungce.robotCleanerOperatingState", ): (["charging", "paused", "cleaning"], )
    },
    'airPurifier': {
        ("switch", ): (["on", "off"], ),
        ("airConditionerFanMode", ): (["smart", "max", "windfree", "sleep"], ),
        ("custom.periodicSensing_periodicSensing", ): (["on", "off"], ),
        ("custom.periodicSensing_periodicSensingInterval", ): (range(600, 3001), ),
        ("custom.periodicSensing_automaticExecutionMode", ): (['airpurify', 'alarm'], ),
        ("custom.periodicSensing_automaticExecutionSetting", ): (["on", "off"], ),
        ("custom.doNotDisturbMode", ): (["on", "off"], ),
    }
}