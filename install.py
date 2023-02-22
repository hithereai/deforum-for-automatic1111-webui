import launch
import os
import sys

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if not launch.is_installed(lib):
            if lib == 'rich' and 'google.colab' in sys.modules:
                launch.run(f'"{sys.executable}" -m pip install {lib} --no-dependencies', desc=f"Installing Deforum requirement: {lib}", errdesc=f"Couldn't install {lib}")
            else:
                launch.run_pip(f"install {lib}", f"Deforum requirement: {lib}")
