#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.1.3),
    on October 15, 2023, at 18:27
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
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2023.1.3'
expName = 'untitled'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
}
# --- Show participant info dialog --
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='F:\\Dhrumna\\Ahmedabad University\\Fourth year\\PSY\\MOTOR SEQUENCE LEARNING\\untitled_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=(1024, 768), fullscr=True, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    backgroundImage='', backgroundFit='none',
    blendMode='avg', useFBO=True, 
    units='height')
win.mouseVisible = False
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
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

# --- Initialize components for Routine "trial" ---
fixation = visual.ShapeStim(
    win=win, name='fixation', vertices='cross',units='pix', 
    size=[1.0, 1.0],
    ori=1.0, pos=[0,0], anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=0.0, interpolate=True)
fixation_2_ = visual.ShapeStim(
    win=win, name='fixation_2_', vertices='cross',units='pix', 
    size=[1.0, 1.0],
    ori=1.0, pos=[0,0], anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
line1 = visual.Rect(
    win=win, name='line1',units='pix', 
    width=(100, 10)[0], height=(100, 10)[1],
    ori=90.0, pos=(-150, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-2.0, interpolate=True)
line2 = visual.Rect(
    win=win, name='line2',units='pix', 
    width=(100, 10)[0], height=(100, 10)[1],
    ori=90.0, pos=(-50, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-3.0, interpolate=True)
line3 = visual.Rect(
    win=win, name='line3',units='pix', 
    width=(100, 10)[0], height=(100, 10)[1],
    ori=90.0, pos=(50, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-4.0, interpolate=True)
line4 = visual.Rect(
    win=win, name='line4',units='pix', 
    width=(100, 10)[0], height=(100, 10)[1],
    ori=90.0, pos=(150, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-5.0, interpolate=True)
probe = visual.ShapeStim(
    win=win, name='probe',units='pix', 
    size=(10, 10), vertices='triangle',
    ori=0.0, pos=[0,0], anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-6.0, interpolate=True)
key_resp = keyboard.Keyboard()

# --- Initialize components for Routine "trial" ---
fixation = visual.ShapeStim(
    win=win, name='fixation', vertices='cross',units='pix', 
    size=[1.0, 1.0],
    ori=1.0, pos=[0,0], anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=0.0, interpolate=True)
fixation_2_ = visual.ShapeStim(
    win=win, name='fixation_2_', vertices='cross',units='pix', 
    size=[1.0, 1.0],
    ori=1.0, pos=[0,0], anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
line1 = visual.Rect(
    win=win, name='line1',units='pix', 
    width=(100, 10)[0], height=(100, 10)[1],
    ori=90.0, pos=(-150, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-2.0, interpolate=True)
line2 = visual.Rect(
    win=win, name='line2',units='pix', 
    width=(100, 10)[0], height=(100, 10)[1],
    ori=90.0, pos=(-50, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-3.0, interpolate=True)
line3 = visual.Rect(
    win=win, name='line3',units='pix', 
    width=(100, 10)[0], height=(100, 10)[1],
    ori=90.0, pos=(50, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-4.0, interpolate=True)
line4 = visual.Rect(
    win=win, name='line4',units='pix', 
    width=(100, 10)[0], height=(100, 10)[1],
    ori=90.0, pos=(150, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-5.0, interpolate=True)
probe = visual.ShapeStim(
    win=win, name='probe',units='pix', 
    size=(10, 10), vertices='triangle',
    ori=0.0, pos=[0,0], anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-6.0, interpolate=True)
key_resp = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=50.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('Book1.xlsx'),
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec('{} = thisTrial[paramName]'.format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    # --- Prepare to start Routine "trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    fixation.setPos((0, 0))
    fixation.setSize((10, 10))
    fixation.setOri(0.0)
    fixation_2_.setPos((0, 0))
    fixation_2_.setSize((10, 10))
    fixation_2_.setOri(0.0)
    probe.setPos((pos, 60))
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    trialComponents = [fixation, fixation_2_, line1, line2, line3, line4, probe, key_resp]
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
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation* updates
        
        # if fixation is starting this frame...
        if fixation.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            fixation.frameNStart = frameN  # exact frame index
            fixation.tStart = t  # local t and not account for scr refresh
            fixation.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fixation.started')
            # update status
            fixation.status = STARTED
            fixation.setAutoDraw(True)
        
        # if fixation is active this frame...
        if fixation.status == STARTED:
            # update params
            pass
        
        # if fixation is stopping this frame...
        if fixation.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fixation.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                fixation.tStop = t  # not accounting for scr refresh
                fixation.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.stopped')
                # update status
                fixation.status = FINISHED
                fixation.setAutoDraw(False)
        
        # *fixation_2_* updates
        
        # if fixation_2_ is starting this frame...
        if fixation_2_.status == NOT_STARTED and tThisFlip >= 0.7-frameTolerance:
            # keep track of start time/frame for later
            fixation_2_.frameNStart = frameN  # exact frame index
            fixation_2_.tStart = t  # local t and not account for scr refresh
            fixation_2_.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_2_, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fixation_2_.started')
            # update status
            fixation_2_.status = STARTED
            fixation_2_.setAutoDraw(True)
        
        # if fixation_2_ is active this frame...
        if fixation_2_.status == STARTED:
            # update params
            pass
        
        # if fixation_2_ is stopping this frame...
        if fixation_2_.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fixation_2_.tStartRefresh + 1.2-frameTolerance:
                # keep track of stop time/frame for later
                fixation_2_.tStop = t  # not accounting for scr refresh
                fixation_2_.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_2_.stopped')
                # update status
                fixation_2_.status = FINISHED
                fixation_2_.setAutoDraw(False)
        
        # *line1* updates
        
        # if line1 is starting this frame...
        if line1.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            line1.frameNStart = frameN  # exact frame index
            line1.tStart = t  # local t and not account for scr refresh
            line1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(line1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'line1.started')
            # update status
            line1.status = STARTED
            line1.setAutoDraw(True)
        
        # if line1 is active this frame...
        if line1.status == STARTED:
            # update params
            pass
        
        # *line2* updates
        
        # if line2 is starting this frame...
        if line2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            line2.frameNStart = frameN  # exact frame index
            line2.tStart = t  # local t and not account for scr refresh
            line2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(line2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'line2.started')
            # update status
            line2.status = STARTED
            line2.setAutoDraw(True)
        
        # if line2 is active this frame...
        if line2.status == STARTED:
            # update params
            pass
        
        # *line3* updates
        
        # if line3 is starting this frame...
        if line3.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            line3.frameNStart = frameN  # exact frame index
            line3.tStart = t  # local t and not account for scr refresh
            line3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(line3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'line3.started')
            # update status
            line3.status = STARTED
            line3.setAutoDraw(True)
        
        # if line3 is active this frame...
        if line3.status == STARTED:
            # update params
            pass
        
        # *line4* updates
        
        # if line4 is starting this frame...
        if line4.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            line4.frameNStart = frameN  # exact frame index
            line4.tStart = t  # local t and not account for scr refresh
            line4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(line4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'line4.started')
            # update status
            line4.status = STARTED
            line4.setAutoDraw(True)
        
        # if line4 is active this frame...
        if line4.status == STARTED:
            # update params
            pass
        
        # *probe* updates
        
        # if probe is starting this frame...
        if probe.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
            # keep track of start time/frame for later
            probe.frameNStart = frameN  # exact frame index
            probe.tStart = t  # local t and not account for scr refresh
            probe.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(probe, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'probe.started')
            # update status
            probe.status = STARTED
            probe.setAutoDraw(True)
        
        # if probe is active this frame...
        if probe.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
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
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=None, waitRelease=False)
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
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
            if eyetracker:
                eyetracker.setConnectionState(False)
        
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
    # the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 50.0 repeats of 'trials'


# set up handler to look after randomisation of conditions etc
trials_2 = data.TrialHandler(nReps=50.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('Book1.xlsx'),
    seed=None, name='trials_2')
thisExp.addLoop(trials_2)  # add the loop to the experiment
thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
if thisTrial_2 != None:
    for paramName in thisTrial_2:
        exec('{} = thisTrial_2[paramName]'.format(paramName))

for thisTrial_2 in trials_2:
    currentLoop = trials_2
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            exec('{} = thisTrial_2[paramName]'.format(paramName))
    
    # --- Prepare to start Routine "trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    fixation.setPos((0, 0))
    fixation.setSize((10, 10))
    fixation.setOri(0.0)
    fixation_2_.setPos((0, 0))
    fixation_2_.setSize((10, 10))
    fixation_2_.setOri(0.0)
    probe.setPos((pos, 60))
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    trialComponents = [fixation, fixation_2_, line1, line2, line3, line4, probe, key_resp]
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
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixation* updates
        
        # if fixation is starting this frame...
        if fixation.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            fixation.frameNStart = frameN  # exact frame index
            fixation.tStart = t  # local t and not account for scr refresh
            fixation.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fixation.started')
            # update status
            fixation.status = STARTED
            fixation.setAutoDraw(True)
        
        # if fixation is active this frame...
        if fixation.status == STARTED:
            # update params
            pass
        
        # if fixation is stopping this frame...
        if fixation.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fixation.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                fixation.tStop = t  # not accounting for scr refresh
                fixation.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.stopped')
                # update status
                fixation.status = FINISHED
                fixation.setAutoDraw(False)
        
        # *fixation_2_* updates
        
        # if fixation_2_ is starting this frame...
        if fixation_2_.status == NOT_STARTED and tThisFlip >= 0.7-frameTolerance:
            # keep track of start time/frame for later
            fixation_2_.frameNStart = frameN  # exact frame index
            fixation_2_.tStart = t  # local t and not account for scr refresh
            fixation_2_.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(fixation_2_, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'fixation_2_.started')
            # update status
            fixation_2_.status = STARTED
            fixation_2_.setAutoDraw(True)
        
        # if fixation_2_ is active this frame...
        if fixation_2_.status == STARTED:
            # update params
            pass
        
        # if fixation_2_ is stopping this frame...
        if fixation_2_.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > fixation_2_.tStartRefresh + 1.2-frameTolerance:
                # keep track of stop time/frame for later
                fixation_2_.tStop = t  # not accounting for scr refresh
                fixation_2_.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_2_.stopped')
                # update status
                fixation_2_.status = FINISHED
                fixation_2_.setAutoDraw(False)
        
        # *line1* updates
        
        # if line1 is starting this frame...
        if line1.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            line1.frameNStart = frameN  # exact frame index
            line1.tStart = t  # local t and not account for scr refresh
            line1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(line1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'line1.started')
            # update status
            line1.status = STARTED
            line1.setAutoDraw(True)
        
        # if line1 is active this frame...
        if line1.status == STARTED:
            # update params
            pass
        
        # *line2* updates
        
        # if line2 is starting this frame...
        if line2.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            line2.frameNStart = frameN  # exact frame index
            line2.tStart = t  # local t and not account for scr refresh
            line2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(line2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'line2.started')
            # update status
            line2.status = STARTED
            line2.setAutoDraw(True)
        
        # if line2 is active this frame...
        if line2.status == STARTED:
            # update params
            pass
        
        # *line3* updates
        
        # if line3 is starting this frame...
        if line3.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            line3.frameNStart = frameN  # exact frame index
            line3.tStart = t  # local t and not account for scr refresh
            line3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(line3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'line3.started')
            # update status
            line3.status = STARTED
            line3.setAutoDraw(True)
        
        # if line3 is active this frame...
        if line3.status == STARTED:
            # update params
            pass
        
        # *line4* updates
        
        # if line4 is starting this frame...
        if line4.status == NOT_STARTED and tThisFlip >= 1-frameTolerance:
            # keep track of start time/frame for later
            line4.frameNStart = frameN  # exact frame index
            line4.tStart = t  # local t and not account for scr refresh
            line4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(line4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'line4.started')
            # update status
            line4.status = STARTED
            line4.setAutoDraw(True)
        
        # if line4 is active this frame...
        if line4.status == STARTED:
            # update params
            pass
        
        # *probe* updates
        
        # if probe is starting this frame...
        if probe.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
            # keep track of start time/frame for later
            probe.frameNStart = frameN  # exact frame index
            probe.tStart = t  # local t and not account for scr refresh
            probe.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(probe, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'probe.started')
            # update status
            probe.status = STARTED
            probe.setAutoDraw(True)
        
        # if probe is active this frame...
        if probe.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
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
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=None, waitRelease=False)
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
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
            if eyetracker:
                eyetracker.setConnectionState(False)
        
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
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
        # was no response the correct answer?!
        if str(corr_resp).lower() == 'none':
           key_resp.corr = 1;  # correct non-response
        else:
           key_resp.corr = 0;  # failed to respond (incorrectly)
    # store data for trials_2 (TrialHandler)
    trials_2.addData('key_resp.keys',key_resp.keys)
    trials_2.addData('key_resp.corr', key_resp.corr)
    if key_resp.keys != None:  # we had a response
        trials_2.addData('key_resp.rt', key_resp.rt)
        trials_2.addData('key_resp.duration', key_resp.duration)
    # the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 50.0 repeats of 'trials_2'


# --- End experiment ---
# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
