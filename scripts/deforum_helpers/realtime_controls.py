# import tempfile
import os
from types import SimpleNamespace
from .args import RealTimeControlArgs

rt_control_args = SimpleNamespace(**RealTimeControlArgs())

frame_path = rt_control_args.frame_path
frame_lockfile_path = rt_control_args.frame_lockfile_path
prompt_path = rt_control_args.prompt_path
deforumSettingsLockFilePath = rt_control_args.deforumSettingsLockFilePath

def lock():
    try:
        with open(deforumSettingsLockFilePath, 'x') as lockfile:
            # write the PID of the current process so you can debug
            # later if a lockfile can be deleted after a program crash
            lockfile.write(str(os.getpid()))
            lockfile.close()
            return True
    except IOError:
         # file already exists
        #print("ALREADY LOCKED")
        return False
        
def unlock():
    os.remove(deforumSettingsLockFilePath)

def lock_frame():
    try:
        with open(frame_lockfile_path, 'x') as lockfile:
            # write the PID of the current process so you can debug
            # later if a lockfile can be deleted after a program crash
            lockfile.write(str(os.getpid()))
            lockfile.close()
            return True
    except IOError:
         # file already exists
        #print("ALREADY LOCKED")
        return False
def unlock_frame():
    os.remove(frame_lockfile_path)

#END OF FULHACK