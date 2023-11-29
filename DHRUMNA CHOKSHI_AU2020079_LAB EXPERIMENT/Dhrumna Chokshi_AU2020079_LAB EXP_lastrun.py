#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on November 29, 2023, at 19:38
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'untitledFINAL'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Dhrumna\\Ahmedabad University\\Fourth year\\PSY PROJECT FINAL\\Dhrumna Chokshi_AU2020079_LAB EXP_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=(1024, 768), fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "trial" ---
    text = visual.TextStim(win=win, name='text',
        text='HELLO!!\n\nThis experiment will take only few minutes. It would be great if you give all your attention and participate',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 0.8824], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_2 = visual.TextStim(win=win, name='text_2',
        text="Are you ready?\n\nLet's Get Started !\n",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=[1.0000, 1.0000, 0.8824], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_3 = visual.TextStim(win=win, name='text_3',
        text='Get ready to have a glimpse of some beautiful pictures :)',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "all_pictures" ---
    blackroad = visual.ImageStim(
        win=win,
        name='blackroad', 
        image='C:/Dhrumna/Ahmedabad University/Fourth year/PSY PROJECT/Images/Dark image 2.JPG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    yellowillusion = visual.ImageStim(
        win=win,
        name='yellowillusion', 
        image='C:/Dhrumna/Ahmedabad University/Fourth year/PSY PROJECT/Images/Yellow blue image.JPG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    redillusion = visual.ImageStim(
        win=win,
        name='redillusion', 
        image='C:/Dhrumna/Ahmedabad University/Fourth year/PSY PROJECT/Images/Red image.JPG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    wall = visual.ImageStim(
        win=win,
        name='wall', 
        image='C:/Dhrumna/Ahmedabad University/Fourth year/PSY PROJECT/Images/wall.JPG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='C:/Dhrumna/Ahmedabad University/Fourth year/PSY PROJECT/Images/colourful new.JPG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    
    # --- Initialize components for Routine "instructions2" ---
    instructions = visual.TextStim(win=win, name='instructions',
        text='I hope you love the pictures :)\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_4 = visual.TextStim(win=win, name='text_4',
        text='The pictures you saw earlier will now be displayed twice. \n\nREAD the upcoming instructions carefully!',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_5 = visual.TextStim(win=win, name='text_5',
        text="If you remember the picture press 'r' key on your keyboard\n\nIf you do not remember the picture press 'n' key on your keyboard\n\n",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_6 = visual.TextStim(win=win, name='text_6',
        text="You will see 2 pictures and then those 2 pictures will be Repeated.\n\nIn both the cases you have to press 'r' or 'n' key by default.  \n\n\nDO NOT PRESS ANY OTHER KEY!",
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "Routine_1" ---
    yellowillusionimage = visual.ImageStim(
        win=win,
        name='yellowillusionimage', 
        image='C:/Dhrumna/Ahmedabad University/Fourth year/PSY PROJECT/Images/Yellow blue image.JPG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    blackroadimage = visual.ImageStim(
        win=win,
        name='blackroadimage', 
        image='C:/Dhrumna/Ahmedabad University/Fourth year/PSY PROJECT/Images/Dark image 2.JPG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "routine_2" ---
    pastelimage = visual.ImageStim(
        win=win,
        name='pastelimage', 
        image='C:/Dhrumna/Ahmedabad University/Fourth year/PSY PROJECT/Images/In house pastel image.JPG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    redillusionimage = visual.ImageStim(
        win=win,
        name='redillusionimage', 
        image='C:/Dhrumna/Ahmedabad University/Fourth year/PSY PROJECT/Images/Red image.JPG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    kr = keyboard.Keyboard()
    
    # --- Initialize components for Routine "routine_3" ---
    whiteimage = visual.ImageStim(
        win=win,
        name='whiteimage', 
        image='C:/Dhrumna/Ahmedabad University/Fourth year/PSY PROJECT/Images/white image.JPG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    yellowwhiteimage = visual.ImageStim(
        win=win,
        name='yellowwhiteimage', 
        image='C:/Dhrumna/Ahmedabad University/Fourth year/PSY PROJECT/Images/yellow-white.JPG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    kr_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "routine_4" ---
    colourfulimage = visual.ImageStim(
        win=win,
        name='colourfulimage', 
        image='C:/Dhrumna/Ahmedabad University/Fourth year/PSY PROJECT/Images/colourful new.JPG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    wallimage = visual.ImageStim(
        win=win,
        name='wallimage', 
        image='C:/Dhrumna/Ahmedabad University/Fourth year/PSY PROJECT/Images/wall.JPG', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    kr_3 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Thanks" ---
    text_7 = visual.TextStim(win=win, name='text_7',
        text='THANK YOU :)\n',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('trial.started', globalClock.getTime())
    # keep track of which components have finished
    trialComponents = [text, text_2, text_3]
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "trial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 15.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 8-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 8-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # if text_2 is stopping this frame...
        if text_2.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 10-frameTolerance:
                # keep track of stop time/frame for later
                text_2.tStop = t  # not accounting for scr refresh
                text_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.stopped')
                # update status
                text_2.status = FINISHED
                text_2.setAutoDraw(False)
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 10-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # if text_3 is stopping this frame...
        if text_3.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 15-frameTolerance:
                # keep track of stop time/frame for later
                text_3.tStop = t  # not accounting for scr refresh
                text_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.stopped')
                # update status
                text_3.status = FINISHED
                text_3.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "trial" ---
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('trial.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-15.000000)
    
    # --- Prepare to start Routine "all_pictures" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('all_pictures.started', globalClock.getTime())
    # keep track of which components have finished
    all_picturesComponents = [blackroad, yellowillusion, redillusion, wall, image]
    for thisComponent in all_picturesComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "all_pictures" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 15.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *blackroad* updates
        
        # if blackroad is starting this frame...
        if blackroad.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            blackroad.frameNStart = frameN  # exact frame index
            blackroad.tStart = t  # local t and not account for scr refresh
            blackroad.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(blackroad, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'blackroad.started')
            # update status
            blackroad.status = STARTED
            blackroad.setAutoDraw(True)
        
        # if blackroad is active this frame...
        if blackroad.status == STARTED:
            # update params
            pass
        
        # if blackroad is stopping this frame...
        if blackroad.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 3-frameTolerance:
                # keep track of stop time/frame for later
                blackroad.tStop = t  # not accounting for scr refresh
                blackroad.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blackroad.stopped')
                # update status
                blackroad.status = FINISHED
                blackroad.setAutoDraw(False)
        
        # *yellowillusion* updates
        
        # if yellowillusion is starting this frame...
        if yellowillusion.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            yellowillusion.frameNStart = frameN  # exact frame index
            yellowillusion.tStart = t  # local t and not account for scr refresh
            yellowillusion.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(yellowillusion, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'yellowillusion.started')
            # update status
            yellowillusion.status = STARTED
            yellowillusion.setAutoDraw(True)
        
        # if yellowillusion is active this frame...
        if yellowillusion.status == STARTED:
            # update params
            pass
        
        # if yellowillusion is stopping this frame...
        if yellowillusion.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 6-frameTolerance:
                # keep track of stop time/frame for later
                yellowillusion.tStop = t  # not accounting for scr refresh
                yellowillusion.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yellowillusion.stopped')
                # update status
                yellowillusion.status = FINISHED
                yellowillusion.setAutoDraw(False)
        
        # *redillusion* updates
        
        # if redillusion is starting this frame...
        if redillusion.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
            # keep track of start time/frame for later
            redillusion.frameNStart = frameN  # exact frame index
            redillusion.tStart = t  # local t and not account for scr refresh
            redillusion.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(redillusion, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'redillusion.started')
            # update status
            redillusion.status = STARTED
            redillusion.setAutoDraw(True)
        
        # if redillusion is active this frame...
        if redillusion.status == STARTED:
            # update params
            pass
        
        # if redillusion is stopping this frame...
        if redillusion.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 9-frameTolerance:
                # keep track of stop time/frame for later
                redillusion.tStop = t  # not accounting for scr refresh
                redillusion.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'redillusion.stopped')
                # update status
                redillusion.status = FINISHED
                redillusion.setAutoDraw(False)
        
        # *wall* updates
        
        # if wall is starting this frame...
        if wall.status == NOT_STARTED and tThisFlip >= 9-frameTolerance:
            # keep track of start time/frame for later
            wall.frameNStart = frameN  # exact frame index
            wall.tStart = t  # local t and not account for scr refresh
            wall.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(wall, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'wall.started')
            # update status
            wall.status = STARTED
            wall.setAutoDraw(True)
        
        # if wall is active this frame...
        if wall.status == STARTED:
            # update params
            pass
        
        # if wall is stopping this frame...
        if wall.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 12-frameTolerance:
                # keep track of stop time/frame for later
                wall.tStop = t  # not accounting for scr refresh
                wall.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wall.stopped')
                # update status
                wall.status = FINISHED
                wall.setAutoDraw(False)
        
        # *image* updates
        
        # if image is starting this frame...
        if image.status == NOT_STARTED and tThisFlip >= 12-frameTolerance:
            # keep track of start time/frame for later
            image.frameNStart = frameN  # exact frame index
            image.tStart = t  # local t and not account for scr refresh
            image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image.started')
            # update status
            image.status = STARTED
            image.setAutoDraw(True)
        
        # if image is active this frame...
        if image.status == STARTED:
            # update params
            pass
        
        # if image is stopping this frame...
        if image.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 15-frameTolerance:
                # keep track of stop time/frame for later
                image.tStop = t  # not accounting for scr refresh
                image.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.stopped')
                # update status
                image.status = FINISHED
                image.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in all_picturesComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "all_pictures" ---
    for thisComponent in all_picturesComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('all_pictures.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-15.000000)
    
    # --- Prepare to start Routine "instructions2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions2.started', globalClock.getTime())
    # keep track of which components have finished
    instructions2Components = [instructions, text_4, text_5, text_6]
    for thisComponent in instructions2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 33.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions* updates
        
        # if instructions is starting this frame...
        if instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions.frameNStart = frameN  # exact frame index
            instructions.tStart = t  # local t and not account for scr refresh
            instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions.started')
            # update status
            instructions.status = STARTED
            instructions.setAutoDraw(True)
        
        # if instructions is active this frame...
        if instructions.status == STARTED:
            # update params
            pass
        
        # if instructions is stopping this frame...
        if instructions.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 3-frameTolerance:
                # keep track of stop time/frame for later
                instructions.tStop = t  # not accounting for scr refresh
                instructions.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instructions.stopped')
                # update status
                instructions.status = FINISHED
                instructions.setAutoDraw(False)
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # if text_4 is stopping this frame...
        if text_4.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 13-frameTolerance:
                # keep track of stop time/frame for later
                text_4.tStop = t  # not accounting for scr refresh
                text_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_4.stopped')
                # update status
                text_4.status = FINISHED
                text_4.setAutoDraw(False)
        
        # *text_5* updates
        
        # if text_5 is starting this frame...
        if text_5.status == NOT_STARTED and tThisFlip >= 13-frameTolerance:
            # keep track of start time/frame for later
            text_5.frameNStart = frameN  # exact frame index
            text_5.tStart = t  # local t and not account for scr refresh
            text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_5.started')
            # update status
            text_5.status = STARTED
            text_5.setAutoDraw(True)
        
        # if text_5 is active this frame...
        if text_5.status == STARTED:
            # update params
            pass
        
        # if text_5 is stopping this frame...
        if text_5.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 23-frameTolerance:
                # keep track of stop time/frame for later
                text_5.tStop = t  # not accounting for scr refresh
                text_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_5.stopped')
                # update status
                text_5.status = FINISHED
                text_5.setAutoDraw(False)
        
        # *text_6* updates
        
        # if text_6 is starting this frame...
        if text_6.status == NOT_STARTED and tThisFlip >= 23-frameTolerance:
            # keep track of start time/frame for later
            text_6.frameNStart = frameN  # exact frame index
            text_6.tStart = t  # local t and not account for scr refresh
            text_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_6.started')
            # update status
            text_6.status = STARTED
            text_6.setAutoDraw(True)
        
        # if text_6 is active this frame...
        if text_6.status == STARTED:
            # update params
            pass
        
        # if text_6 is stopping this frame...
        if text_6.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 33-frameTolerance:
                # keep track of stop time/frame for later
                text_6.tStop = t  # not accounting for scr refresh
                text_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_6.stopped')
                # update status
                text_6.status = FINISHED
                text_6.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions2" ---
    for thisComponent in instructions2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions2.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-33.000000)
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('Dhrumna Chokshi_AU2020079 _CONDITION EXCEL SHEETS/Routine1.xlsx'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Routine_1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Routine_1.started', globalClock.getTime())
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # keep track of which components have finished
        Routine_1Components = [yellowillusionimage, blackroadimage, key_resp]
        for thisComponent in Routine_1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Routine_1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 6.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *yellowillusionimage* updates
            
            # if yellowillusionimage is starting this frame...
            if yellowillusionimage.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                yellowillusionimage.frameNStart = frameN  # exact frame index
                yellowillusionimage.tStart = t  # local t and not account for scr refresh
                yellowillusionimage.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(yellowillusionimage, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yellowillusionimage.started')
                # update status
                yellowillusionimage.status = STARTED
                yellowillusionimage.setAutoDraw(True)
            
            # if yellowillusionimage is active this frame...
            if yellowillusionimage.status == STARTED:
                # update params
                pass
            
            # if yellowillusionimage is stopping this frame...
            if yellowillusionimage.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 3-frameTolerance:
                    # keep track of stop time/frame for later
                    yellowillusionimage.tStop = t  # not accounting for scr refresh
                    yellowillusionimage.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'yellowillusionimage.stopped')
                    # update status
                    yellowillusionimage.status = FINISHED
                    yellowillusionimage.setAutoDraw(False)
            
            # *blackroadimage* updates
            
            # if blackroadimage is starting this frame...
            if blackroadimage.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                # keep track of start time/frame for later
                blackroadimage.frameNStart = frameN  # exact frame index
                blackroadimage.tStart = t  # local t and not account for scr refresh
                blackroadimage.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blackroadimage, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blackroadimage.started')
                # update status
                blackroadimage.status = STARTED
                blackroadimage.setAutoDraw(True)
            
            # if blackroadimage is active this frame...
            if blackroadimage.status == STARTED:
                # update params
                pass
            
            # if blackroadimage is stopping this frame...
            if blackroadimage.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6-frameTolerance:
                    # keep track of stop time/frame for later
                    blackroadimage.tStop = t  # not accounting for scr refresh
                    blackroadimage.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blackroadimage.stopped')
                    # update status
                    blackroadimage.status = FINISHED
                    blackroadimage.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['r', 'n'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # was this correct?
                    if (key_resp.keys == str(corr_resp)) or (key_resp.keys == corr_resp):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Routine_1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Routine_1" ---
        for thisComponent in Routine_1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Routine_1.stopped', globalClock.getTime())
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(corr_resp).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for trials (TrialHandler)
        trials.addData('key_resp.keys',key_resp.keys)
        trials.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            trials.addData('key_resp.rt', key_resp.rt)
            trials.addData('key_resp.duration', key_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-6.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials'
    
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('Dhrumna Chokshi_AU2020079 _CONDITION EXCEL SHEETS/Routine2.xlsx'),
        seed=None, name='trials_2')
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "routine_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('routine_2.started', globalClock.getTime())
        kr.keys = []
        kr.rt = []
        _kr_allKeys = []
        # keep track of which components have finished
        routine_2Components = [pastelimage, redillusionimage, kr]
        for thisComponent in routine_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "routine_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 6.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *pastelimage* updates
            
            # if pastelimage is starting this frame...
            if pastelimage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pastelimage.frameNStart = frameN  # exact frame index
                pastelimage.tStart = t  # local t and not account for scr refresh
                pastelimage.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pastelimage, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pastelimage.started')
                # update status
                pastelimage.status = STARTED
                pastelimage.setAutoDraw(True)
            
            # if pastelimage is active this frame...
            if pastelimage.status == STARTED:
                # update params
                pass
            
            # if pastelimage is stopping this frame...
            if pastelimage.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 3-frameTolerance:
                    # keep track of stop time/frame for later
                    pastelimage.tStop = t  # not accounting for scr refresh
                    pastelimage.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'pastelimage.stopped')
                    # update status
                    pastelimage.status = FINISHED
                    pastelimage.setAutoDraw(False)
            
            # *redillusionimage* updates
            
            # if redillusionimage is starting this frame...
            if redillusionimage.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                # keep track of start time/frame for later
                redillusionimage.frameNStart = frameN  # exact frame index
                redillusionimage.tStart = t  # local t and not account for scr refresh
                redillusionimage.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(redillusionimage, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'redillusionimage.started')
                # update status
                redillusionimage.status = STARTED
                redillusionimage.setAutoDraw(True)
            
            # if redillusionimage is active this frame...
            if redillusionimage.status == STARTED:
                # update params
                pass
            
            # if redillusionimage is stopping this frame...
            if redillusionimage.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6-frameTolerance:
                    # keep track of stop time/frame for later
                    redillusionimage.tStop = t  # not accounting for scr refresh
                    redillusionimage.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'redillusionimage.stopped')
                    # update status
                    redillusionimage.status = FINISHED
                    redillusionimage.setAutoDraw(False)
            
            # *kr* updates
            waitOnFlip = False
            
            # if kr is starting this frame...
            if kr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                kr.frameNStart = frameN  # exact frame index
                kr.tStart = t  # local t and not account for scr refresh
                kr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(kr, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'kr.started')
                # update status
                kr.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(kr.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(kr.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if kr is stopping this frame...
            if kr.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6-frameTolerance:
                    # keep track of stop time/frame for later
                    kr.tStop = t  # not accounting for scr refresh
                    kr.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'kr.stopped')
                    # update status
                    kr.status = FINISHED
                    kr.status = FINISHED
            if kr.status == STARTED and not waitOnFlip:
                theseKeys = kr.getKeys(keyList=['r', 'n'], ignoreKeys=["escape"], waitRelease=False)
                _kr_allKeys.extend(theseKeys)
                if len(_kr_allKeys):
                    kr.keys = _kr_allKeys[-1].name  # just the last key pressed
                    kr.rt = _kr_allKeys[-1].rt
                    kr.duration = _kr_allKeys[-1].duration
                    # was this correct?
                    if (kr.keys == str(corr_resp)) or (kr.keys == corr_resp):
                        kr.corr = 1
                    else:
                        kr.corr = 0
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in routine_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "routine_2" ---
        for thisComponent in routine_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('routine_2.stopped', globalClock.getTime())
        # check responses
        if kr.keys in ['', [], None]:  # No response was made
            kr.keys = None
            # was no response the correct answer?!
            if str(corr_resp).lower() == 'none':
               kr.corr = 1;  # correct non-response
            else:
               kr.corr = 0;  # failed to respond (incorrectly)
        # store data for trials_2 (TrialHandler)
        trials_2.addData('kr.keys',kr.keys)
        trials_2.addData('kr.corr', kr.corr)
        if kr.keys != None:  # we had a response
            trials_2.addData('kr.rt', kr.rt)
            trials_2.addData('kr.duration', kr.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-6.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_2'
    
    
    # set up handler to look after randomisation of conditions etc
    trials_3 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('Dhrumna Chokshi_AU2020079 _CONDITION EXCEL SHEETS/Routine3.xlsx'),
        seed=None, name='trials_3')
    thisExp.addLoop(trials_3)  # add the loop to the experiment
    thisTrial_3 = trials_3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
    if thisTrial_3 != None:
        for paramName in thisTrial_3:
            globals()[paramName] = thisTrial_3[paramName]
    
    for thisTrial_3 in trials_3:
        currentLoop = trials_3
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
        if thisTrial_3 != None:
            for paramName in thisTrial_3:
                globals()[paramName] = thisTrial_3[paramName]
        
        # --- Prepare to start Routine "routine_3" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('routine_3.started', globalClock.getTime())
        kr_2.keys = []
        kr_2.rt = []
        _kr_2_allKeys = []
        # keep track of which components have finished
        routine_3Components = [whiteimage, yellowwhiteimage, kr_2]
        for thisComponent in routine_3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "routine_3" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 6.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *whiteimage* updates
            
            # if whiteimage is starting this frame...
            if whiteimage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                whiteimage.frameNStart = frameN  # exact frame index
                whiteimage.tStart = t  # local t and not account for scr refresh
                whiteimage.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(whiteimage, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'whiteimage.started')
                # update status
                whiteimage.status = STARTED
                whiteimage.setAutoDraw(True)
            
            # if whiteimage is active this frame...
            if whiteimage.status == STARTED:
                # update params
                pass
            
            # if whiteimage is stopping this frame...
            if whiteimage.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 3-frameTolerance:
                    # keep track of stop time/frame for later
                    whiteimage.tStop = t  # not accounting for scr refresh
                    whiteimage.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'whiteimage.stopped')
                    # update status
                    whiteimage.status = FINISHED
                    whiteimage.setAutoDraw(False)
            
            # *yellowwhiteimage* updates
            
            # if yellowwhiteimage is starting this frame...
            if yellowwhiteimage.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                # keep track of start time/frame for later
                yellowwhiteimage.frameNStart = frameN  # exact frame index
                yellowwhiteimage.tStart = t  # local t and not account for scr refresh
                yellowwhiteimage.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(yellowwhiteimage, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'yellowwhiteimage.started')
                # update status
                yellowwhiteimage.status = STARTED
                yellowwhiteimage.setAutoDraw(True)
            
            # if yellowwhiteimage is active this frame...
            if yellowwhiteimage.status == STARTED:
                # update params
                pass
            
            # if yellowwhiteimage is stopping this frame...
            if yellowwhiteimage.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6-frameTolerance:
                    # keep track of stop time/frame for later
                    yellowwhiteimage.tStop = t  # not accounting for scr refresh
                    yellowwhiteimage.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'yellowwhiteimage.stopped')
                    # update status
                    yellowwhiteimage.status = FINISHED
                    yellowwhiteimage.setAutoDraw(False)
            
            # *kr_2* updates
            waitOnFlip = False
            
            # if kr_2 is starting this frame...
            if kr_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                kr_2.frameNStart = frameN  # exact frame index
                kr_2.tStart = t  # local t and not account for scr refresh
                kr_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(kr_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'kr_2.started')
                # update status
                kr_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(kr_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(kr_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if kr_2 is stopping this frame...
            if kr_2.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6-frameTolerance:
                    # keep track of stop time/frame for later
                    kr_2.tStop = t  # not accounting for scr refresh
                    kr_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'kr_2.stopped')
                    # update status
                    kr_2.status = FINISHED
                    kr_2.status = FINISHED
            if kr_2.status == STARTED and not waitOnFlip:
                theseKeys = kr_2.getKeys(keyList=['r', 'n'], ignoreKeys=["escape"], waitRelease=False)
                _kr_2_allKeys.extend(theseKeys)
                if len(_kr_2_allKeys):
                    kr_2.keys = _kr_2_allKeys[-1].name  # just the last key pressed
                    kr_2.rt = _kr_2_allKeys[-1].rt
                    kr_2.duration = _kr_2_allKeys[-1].duration
                    # was this correct?
                    if (kr_2.keys == str(corr_resp)) or (kr_2.keys == corr_resp):
                        kr_2.corr = 1
                    else:
                        kr_2.corr = 0
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in routine_3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "routine_3" ---
        for thisComponent in routine_3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('routine_3.stopped', globalClock.getTime())
        # check responses
        if kr_2.keys in ['', [], None]:  # No response was made
            kr_2.keys = None
            # was no response the correct answer?!
            if str(corr_resp).lower() == 'none':
               kr_2.corr = 1;  # correct non-response
            else:
               kr_2.corr = 0;  # failed to respond (incorrectly)
        # store data for trials_3 (TrialHandler)
        trials_3.addData('kr_2.keys',kr_2.keys)
        trials_3.addData('kr_2.corr', kr_2.corr)
        if kr_2.keys != None:  # we had a response
            trials_3.addData('kr_2.rt', kr_2.rt)
            trials_3.addData('kr_2.duration', kr_2.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-6.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_3'
    
    
    # set up handler to look after randomisation of conditions etc
    trials_4 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('Dhrumna Chokshi_AU2020079 _CONDITION EXCEL SHEETS/Routine4.xlsx'),
        seed=None, name='trials_4')
    thisExp.addLoop(trials_4)  # add the loop to the experiment
    thisTrial_4 = trials_4.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
    if thisTrial_4 != None:
        for paramName in thisTrial_4:
            globals()[paramName] = thisTrial_4[paramName]
    
    for thisTrial_4 in trials_4:
        currentLoop = trials_4
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
        if thisTrial_4 != None:
            for paramName in thisTrial_4:
                globals()[paramName] = thisTrial_4[paramName]
        
        # --- Prepare to start Routine "routine_4" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('routine_4.started', globalClock.getTime())
        kr_3.keys = []
        kr_3.rt = []
        _kr_3_allKeys = []
        # keep track of which components have finished
        routine_4Components = [colourfulimage, wallimage, kr_3]
        for thisComponent in routine_4Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "routine_4" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 6.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *colourfulimage* updates
            
            # if colourfulimage is starting this frame...
            if colourfulimage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                colourfulimage.frameNStart = frameN  # exact frame index
                colourfulimage.tStart = t  # local t and not account for scr refresh
                colourfulimage.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(colourfulimage, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'colourfulimage.started')
                # update status
                colourfulimage.status = STARTED
                colourfulimage.setAutoDraw(True)
            
            # if colourfulimage is active this frame...
            if colourfulimage.status == STARTED:
                # update params
                pass
            
            # if colourfulimage is stopping this frame...
            if colourfulimage.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 3-frameTolerance:
                    # keep track of stop time/frame for later
                    colourfulimage.tStop = t  # not accounting for scr refresh
                    colourfulimage.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'colourfulimage.stopped')
                    # update status
                    colourfulimage.status = FINISHED
                    colourfulimage.setAutoDraw(False)
            
            # *wallimage* updates
            
            # if wallimage is starting this frame...
            if wallimage.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
                # keep track of start time/frame for later
                wallimage.frameNStart = frameN  # exact frame index
                wallimage.tStart = t  # local t and not account for scr refresh
                wallimage.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(wallimage, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'wallimage.started')
                # update status
                wallimage.status = STARTED
                wallimage.setAutoDraw(True)
            
            # if wallimage is active this frame...
            if wallimage.status == STARTED:
                # update params
                pass
            
            # if wallimage is stopping this frame...
            if wallimage.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6-frameTolerance:
                    # keep track of stop time/frame for later
                    wallimage.tStop = t  # not accounting for scr refresh
                    wallimage.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'wallimage.stopped')
                    # update status
                    wallimage.status = FINISHED
                    wallimage.setAutoDraw(False)
            
            # *kr_3* updates
            waitOnFlip = False
            
            # if kr_3 is starting this frame...
            if kr_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                kr_3.frameNStart = frameN  # exact frame index
                kr_3.tStart = t  # local t and not account for scr refresh
                kr_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(kr_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'kr_3.started')
                # update status
                kr_3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(kr_3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(kr_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if kr_3 is stopping this frame...
            if kr_3.status == STARTED:
                # is it time to stop? (based on local clock)
                if tThisFlip > 6-frameTolerance:
                    # keep track of stop time/frame for later
                    kr_3.tStop = t  # not accounting for scr refresh
                    kr_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'kr_3.stopped')
                    # update status
                    kr_3.status = FINISHED
                    kr_3.status = FINISHED
            if kr_3.status == STARTED and not waitOnFlip:
                theseKeys = kr_3.getKeys(keyList=['r', 'n'], ignoreKeys=["escape"], waitRelease=False)
                _kr_3_allKeys.extend(theseKeys)
                if len(_kr_3_allKeys):
                    kr_3.keys = _kr_3_allKeys[-1].name  # just the last key pressed
                    kr_3.rt = _kr_3_allKeys[-1].rt
                    kr_3.duration = _kr_3_allKeys[-1].duration
                    # was this correct?
                    if (kr_3.keys == str(corr_resp)) or (kr_3.keys == corr_resp):
                        kr_3.corr = 1
                    else:
                        kr_3.corr = 0
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in routine_4Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "routine_4" ---
        for thisComponent in routine_4Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('routine_4.stopped', globalClock.getTime())
        # check responses
        if kr_3.keys in ['', [], None]:  # No response was made
            kr_3.keys = None
            # was no response the correct answer?!
            if str(corr_resp).lower() == 'none':
               kr_3.corr = 1;  # correct non-response
            else:
               kr_3.corr = 0;  # failed to respond (incorrectly)
        # store data for trials_4 (TrialHandler)
        trials_4.addData('kr_3.keys',kr_3.keys)
        trials_4.addData('kr_3.corr', kr_3.corr)
        if kr_3.keys != None:  # we had a response
            trials_4.addData('kr_3.rt', kr_3.rt)
            trials_4.addData('kr_3.duration', kr_3.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-6.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_4'
    
    
    # --- Prepare to start Routine "Thanks" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Thanks.started', globalClock.getTime())
    # keep track of which components have finished
    ThanksComponents = [text_7]
    for thisComponent in ThanksComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Thanks" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_7* updates
        
        # if text_7 is starting this frame...
        if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_7.frameNStart = frameN  # exact frame index
            text_7.tStart = t  # local t and not account for scr refresh
            text_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_7.started')
            # update status
            text_7.status = STARTED
            text_7.setAutoDraw(True)
        
        # if text_7 is active this frame...
        if text_7.status == STARTED:
            # update params
            pass
        
        # if text_7 is stopping this frame...
        if text_7.status == STARTED:
            # is it time to stop? (based on local clock)
            if tThisFlip > 3-frameTolerance:
                # keep track of stop time/frame for later
                text_7.tStop = t  # not accounting for scr refresh
                text_7.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_7.stopped')
                # update status
                text_7.status = FINISHED
                text_7.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ThanksComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Thanks" ---
    for thisComponent in ThanksComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Thanks.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
