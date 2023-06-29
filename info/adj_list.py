def bulb_adj_list(state):
    
    # switch off
    if state[('switch',)] == ('on',):
        st = {k:state[k] for k in state}
        st[('switch',)] = ('off',)
        yield st, f'switch off'
    
    # switch on
    if state[('switch',)] == ('off',):
        st = {k:state[k] for k in state}
        st[('switch',)] = ('on',)
        yield st, 'switch on'

    # setLevel
    if state[('switch',)] == ('on',):
        st = {k:state[k] for k in state}
        st[('switch',)] = ('off',)
        yield st, f'setLevel:0'
    
    for arg in range(1, 101):
        if st[('switchLevel',)] == (arg,):
            continue
        st = {k:state[k] for k in state}
        st[('switchLevel',)] = (arg,)
        st[('switch',)] = ('on',)
        yield st, f'setLevel:{arg}'

    # setColorTemperature
    for arg in range(2200, 8901, 100):
        if st[('colorTemperature',)] == (arg,):
            continue
        st = {k:state[k] for k in state}
        st[('colorTemperature',)] = (arg,)
        st[('switch',)] = ('on',)
        yield st, f'setColorTemperature:{arg}'

    # setHue
    for arg in range(0, 101):
        if st[('colorControl_hue',)] == (arg,):
            continue
        st = {k:state[k] for k in state}
        st[('colorControl_hue',)] = (arg,)
        st[('colorControl_saturation',)] = (0,)
        st[('switch',)] = ('on',)
        yield st, f'setHue:{arg}'

    # setSaturation
    for arg in range(0, 101):
        if st[('colorControl_saturation',)] == (arg,):
            continue
        st = {k:state[k] for k in state}
        st[('colorControl_hue',)] = (0,)
        st[('colorControl_saturation',)] = (arg,)
        st[('switch',)] = ('on',)
        yield st, f'setSaturation:{arg}'


def tv_adj_list(state):
    
    # switch
    if state[('switch',)] == ('off',):
        st = {k:state[k] for k in state}
        st[('switch',)] = ('on',)
        # st[('audioMute',)] = ('unmuted',)
        yield st, f'switch on'
    if state[('switch',)] == ('on',):
        st = {k:state[k] for k in state}
        st[('switch',)] = ('off',)
        yield st, f'switch off'
        
    #setVolume
    if state[('switch',)] == ('on',):
        for arg in range(1, 101):
            if state[('audioVolume',)] == (arg,):
                continue
            st = {k:state[k] for k in state}
            st[('audioVolume',)] = (arg,)
            yield st, f'setVolume:{arg}'
            
    #volumeUp
    if state[('switch',)] == ('on',):
        current_volume=state[('audioVolume',)][0]
        if current_volume + 1 <=100:
            st = {k:state[k] for k in state}
            st[('audioVolume',)]=(current_volume+1,)
            yield st, f'volumeUp'
            
    # volumeDown
    if state[('switch',)] == ('on',) :
        current_volume=state[('audioVolume',)][0]
        if current_volume -1 >=0:
            st = {k:state[k] for k in state}
            st[('audioVolume',)]=(current_volume-1,)
            yield st, f'volumeDown'
            
    # channel
    channels= [0, 505, 506, 508, 516, 517, 518, 520, 524, 525, 526, 530, 534, 580, 581, 642, 643, 644, 645, 648, 651, 655, 675, 700, 731, 751, 752, 801, 831, 832, 833, 834, 835, 839]
    
    # channelUp
    if state[('switch',)] == ('on',) and state[('mediaInputSource',)] == ('digitalTv',):
        current_channel_index=channels.index(state[('tvChannel',)][0])
        if current_channel_index +1  <= len(channels) -1:
            st = {k:state[k] for k in state}
            st[('tvChannel',)] = (channels[current_channel_index +1],)
            yield st, f'channelUp'
            
    # channelDown
    if state[('switch',)] == ('on',) and state[('mediaInputSource',)] == ('digitalTv',):
        current_channel_index=channels.index(state[('tvChannel',)][0])
        if current_channel_index -1 >= 0:
            st = {k:state[k] for k in state}
            st[('tvChannel',)] = (channels[current_channel_index-1],)
            yield st, f'channelDown'
    
    # setTvChannel
    if state[('switch',)] == ('on',) and state[('mediaInputSource',)] == ('digitalTv',):
        for arg in range(len(channels)):
            st = {k:state[k] for k in state}
            if state[('tvChannel',)] == (channels[arg],):
                continue
            st[('tvChannel',)]=(channels[arg],)
            yield st, f'setTvChannel:{channels[arg]}'
    
    # audiomute
    if state[('switch',)] == ('on',) and state[('audioMute',)] == ('unmuted',):
        st = {k:state[k] for k in state}
        st[('audioMute',)] = ('muted',)
        yield st, f'mute'
    if state[('switch',)] == ('on',) and state[('audioMute',)] == ('muted',):
        st = {k:state[k] for k in state}
        st[('audioMute',)] = ('unmuted',)
        yield st, f'unmute'
    
    # setMute
    if state[('switch',)] == ('on',) and state[('audioMute',)] == ('unmuted',):
        st = {k:state[k] for k in state}
        st[('audioMute',)] = ('muted',)
        yield st, f'setMute:muted'
    if state[('switch',)] == ('on',) and state[('audioMute',)] == ('muted',):
        st = {k:state[k] for k in state}
        st[('audioMute',)] = ('unmuted',)
        yield st, f'setMute:unmuted'
    
    # mediaInputSource
    if state[('switch',)] == ('on',) and state[('mediaInputSource',)] == ('HDMI2',):
        st = {k:state[k] for k in state}
        st[('mediaInputSource',)] = ('digitalTv',)
        st[('tvChannel',)] = (505, )
        yield st, f'setInputSource:digitalTv'
    if state[('switch',)] == ('on',) and state[('mediaInputSource',)] == ('digitalTv',):
        st = {k:state[k] for k in state}
        st[('mediaInputSource',)] = ('HDMI2',)
        st[('tvChannel',)] = (0, )
        yield st, f'setInputSource:HDMI2'


def robotcleaner_adj_list(state):
    
    # setVolume
    for arg in range(0, 101,20):
        if state[('audioVolume',)] == (arg,):
            continue
        st = {k:state[k] for k in state}
        st[('audioVolume',)] = (arg,)
        if arg == 0:
            st[('audioMute',)] = ('muted',)
        else:
            st[('audioMute',)] = ('unmuted',)
        yield st, f'setVolume:{arg}'
    
    # volumeUp
    current_volume=state[('audioVolume',)][0]
    if current_volume + 20 <=100:
        st = {k:state[k] for k in state}
        st[('audioVolume',)]=(current_volume+20,) 
        if current_volume+20 > 0:
            st[('audioMute',)] = ('unmuted',)
        yield st, f'volumeUp'
        
    # volumeDown
    current_volume=state[('audioVolume',)][0]
    if current_volume -20 >=0:
        st = {k:state[k] for k in state}
        st[('audioVolume',)]=(current_volume-20,)
        if current_volume-20 < 20:
            st[('audioMute',)] = ('muted',)
        yield st, f'volumeDown'
            
    # audioMute: mute, unmute
    if state[('audioMute',)] == ('unmuted',):
        st = {k:state[k] for k in state}
        st[('audioMute',)] = ('muted',)
        st[('audioVolume',)]=(0,)
        yield st, f'mute'
        
    if state[('audioMute',)] == ('muted',):
        st = {k:state[k] for k in state}
        st[('audioMute',)] = ('unmuted',)
        st[('audioVolume',)]=(80,)
        yield st, f'unmute'
    
    # setMute
    if state[('audioMute',)] == ('unmuted',):
        st = {k:state[k] for k in state}
        st[('audioMute',)] = ('muted',)
        st[('audioVolume',)]=(0,)
        yield st, f'setMute:muted'

    if state[('audioMute',)] == ('muted',):
        st = {k:state[k] for k in state}
        st[('audioMute',)] = ('unmuted',)
        st[('audioVolume',)]=(40,)
        yield st, f'setMute:unmuted'
    
    # setRobotCleanerTurboMode
    robotCleanerTurboMode=["silence","on","off"] 
    for arg in robotCleanerTurboMode:
        st = {k:state[k] for k in state}
        if state[('robotCleanerTurboMode',)] == (arg,):
            continue
        st[('robotCleanerTurboMode',)]=(arg,) 
        yield st, f'setRobotCleanerTurboMode:{arg}'

    # samsungce.robotCleanerOperatingState
    samsungce_robotCleanerOperatingState=["charging","paused","cleaning"]
    
    # robotCleanerMovement
    robotCleanerMovement=["homing","pause","cleaning"]
    for op_state, arg in zip(samsungce_robotCleanerOperatingState, robotCleanerMovement):
        st = {k:state[k] for k in state}
        if state[('samsungce.robotCleanerOperatingState',)] == (op_state,):
            continue
        st[('samsungce.robotCleanerOperatingState',)]=(op_state,)
        yield st, f'setRobotCleanerMovement:{arg}'
    
    # samsungce.robotCleanerOperatingState: pause
    st = {k:state[k] for k in state}
    if state[('samsungce.robotCleanerOperatingState',)] != ("paused",):
        st[('samsungce.robotCleanerOperatingState',)]=("paused",)
        yield st, f'pause'
    
    # samsungce.robotCleanerOperatingState: returnToHome
    st = {k:state[k] for k in state}
    if state[('samsungce.robotCleanerOperatingState',)] != ("charging",):
        st[('samsungce.robotCleanerOperatingState',)]=("charging",)
        yield st, f'returnToHome'

    # samsungce.robotCleanerOperatingState: setCleaningMode
    st = {k:state[k] for k in state}
    if state[('samsungce.robotCleanerOperatingState',)] != ("cleaning",):
        st[('samsungce.robotCleanerOperatingState',)]=("cleaning",)
        yield st, f'setCleaningMode:auto'
    else: # cleaning
        st[('samsungce.robotCleanerOperatingState',)]=("paused",)
        yield st, f'setCleaningMode:stop'
    
    # doNotDisturbOff
    if state[('custom.doNotDisturb',)] == ("on",):
        st[('custom.doNotDisturb',)]=("off",)
        yield st, f'doNotDisturbOff'
        
    # doNotDisturbOn
    if state[('custom.doNotDisturb',)] == ("off",):
        st[('custom.doNotDisturb',)]=("on",)
        yield st, f'doNotDisturbOff'
        
        
def airPurifier_adj_list(state):
    # switch
    if state[('switch',)] == ('off',):
        st = {k:state[k] for k in state}
        st[('switch',)] = ('on',)
        st[('airConditionerFanMode',)]=('smart',)
        yield st, f'switch on'

    if state[('switch',)] == ('on',):
        st = {k:state[k] for k in state}
        st[('switch',)] = ('off',)
        yield st, f'switch off'
        
    # doNotDisturbOff
    if state[('custom.doNotDisturbMode',)] == ("on",):
        st = {k:state[k] for k in state}
        st[('custom.doNotDisturbMode',)]=("off",)
        yield st, f'doNotDisturbOff'
        
    # doNotDisturbOn
    if state[('custom.doNotDisturbMode',)] == ("off",):
        st = {k:state[k] for k in state}
        st[('custom.doNotDisturbMode',)]=("on",)
        yield st, f'doNotDisturbOff'
    
    # setFanMode
    airConditionerFanMode=["smart","max","windfree","sleep"]
    if state[('switch',)] == ('on',):
        for arg in airConditionerFanMode:
            if state[('airConditionerFanMode',)] == (arg,):
                continue
            st = {k:state[k] for k in state}
            st[('airConditionerFanMode',)]=(arg,)
            yield st, f'setFanMode:{arg}'
    
    # setPeriodicSensingInterval
    if state[('switch',)] == ('on',): 
        for arg in range(600,3001):
            if state[('custom.periodicSensing_periodicSensingInterval',)] == (arg,):
                continue
            st = {k:state[k] for k in state}
            st[('custom.periodicSensing_periodicSensingInterval',)]=(arg,)
            yield st, f'setPeriodicSensingInterval:{arg}'
    
    # periodicSensingOn
    if state[('custom.periodicSensing_periodicSensing',)] == ('off',): 
        st = {k:state[k] for k in state}
        st[('custom.periodicSensing_periodicSensing',)]=("on",)
        yield st, f'periodicSensingOn:on'
    
    # periodicSensingOff
    if state[('custom.periodicSensing_periodicSensing',)] == ('on',): 
        st = {k:state[k] for k in state}
        st[('custom.periodicSensing_periodicSensing',)]=("off",)
        yield st, f'periodicSensingOn:off'
        
    # setAutomaticExecutionMode
    executionmode=['airpurify', 'alarm']
    if state[('switch',)] == ('on',): 
        for arg in executionmode:
            st = {k:state[k] for k in state}
            if state[('custom.periodicSensing_automaticExecutionMode',)]!=(arg,):
                st[('custom.periodicSensing_automaticExecutionMode',)]=(arg,)
                yield st, f'setAutomaticExecutionMode:{arg}'
    
    # setPeriodicSensing
    periodicSensing_periodicSensing=["on","off"]
    for sensing in periodicSensing_periodicSensing:
        for interval in range(600,3001):
            if not(state[('custom.periodicSensing_periodicSensing',)]==(sensing,) and state[('custom.periodicSensing_periodicSensingInterval',)]==(interval,)):
                st = {k:state[k] for k in state}
                st[('custom.periodicSensing_periodicSensing',)]=(sensing,)
                st[('custom.periodicSensing_periodicSensingInterval',)]=(interval,)
                yield st, f'setPeriodicSensing:{sensing,interval}'
