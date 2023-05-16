# Detach 'deforum_helpers' from 'scripts' to prevent "No module named 'scripts.deforum_helpers'"  error 
# causing Deforum's tab not show up in some cases when you've might've broken the environment with webui packages updates
import sys, os, shutil

deforum_folder_name = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])

basedirs = [os.getcwd()]
if 'google.colab' in sys.modules:
    basedirs.append('/content/gdrive/MyDrive/sd/stable-diffusion-webui') #hardcode as TheLastBen's colab seems to be the primal source

for basedir in basedirs:
    deforum_paths_to_ensure = [
        os.path.join(deforum_folder_name, 'scripts'),
        os.path.join(deforum_folder_name, 'scripts', 'deforum_helpers', 'src')
        ]

    for deforum_scripts_path_fix in deforum_paths_to_ensure:
        if not deforum_scripts_path_fix in sys.path:
            sys.path.extend([deforum_scripts_path_fix])

# Main deforum stuff
import deforum_helpers.args as deforum_args
import deforum_helpers.settings as deforum_settings
from deforum_helpers.save_images import dump_frames_cache, reset_frames_cache
from deforum_helpers.frame_interpolation import process_video_interpolation

import modules.scripts as wscripts
from modules import script_callbacks, ui_components
import gradio as gr
import json
import traceback

from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from PIL import Image
from deforum_helpers.video_audio_utilities import ffmpeg_stitch_video, make_gifski_gif, handle_imgs_deletion, find_ffmpeg_binary, get_ffmpeg_params
from deforum_helpers.general_utils import get_deforum_version
from deforum_helpers.upscaling import make_upscale_v2
import gc
import numpy as np
import torch
from webui import wrap_gradio_gpu_call
import modules.shared as shared
from modules.shared import opts, cmd_opts, state
from modules.ui import create_output_panel, plaintext_to_html, wrap_gradio_call
from types import SimpleNamespace
from deforum_helpers.subtitle_handler import get_user_values

DEBUG_MODE = opts.data.get("deforum_debug_mode_enabled", False)

def on_ui_settings():
    srt_ui_params = get_user_values()
    section = ('deforum', "Deforum")
    shared.opts.add_option("deforum_keep_3d_models_in_vram", shared.OptionInfo(False, "Keep 3D models in VRAM between runs", gr.Checkbox, {"interactive": True, "visible": True if not (cmd_opts.lowvram or cmd_opts.medvram) else False}, section=section))
    shared.opts.add_option("deforum_enable_persistent_settings", shared.OptionInfo(False, "Keep settings persistent upon relaunch of webui", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("deforum_persistent_settings_path", shared.OptionInfo("models/Deforum/deforum_persistent_settings.txt", "Path for saving your persistent settings file:", section=section))
    shared.opts.add_option("deforum_ffmpeg_location", shared.OptionInfo(find_ffmpeg_binary(), "FFmpeg path/ location", section=section))
    shared.opts.add_option("deforum_ffmpeg_crf", shared.OptionInfo(17, "FFmpeg CRF value", gr.Slider, {"interactive": True, "minimum": 0, "maximum": 51}, section=section))
    shared.opts.add_option("deforum_ffmpeg_preset", shared.OptionInfo('slow', "FFmpeg Preset", gr.Dropdown, {"interactive": True, "choices": ['veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast', 'superfast', 'ultrafast']}, section=section))
    shared.opts.add_option("deforum_debug_mode_enabled", shared.OptionInfo(False, "Enable Dev mode - adds extra reporting in console", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("deforum_save_gen_info_as_srt", shared.OptionInfo(False, "Save an .srt (subtitles) file with the generation info along with each animation", gr.Checkbox, {"interactive": True}, section=section))  
    shared.opts.add_option("deforum_embed_srt", shared.OptionInfo(False, "If .srt file is saved, soft-embed the subtitles into the rendered video file", gr.Checkbox, {"interactive": True}, section=section))  
    shared.opts.add_option("deforum_save_gen_info_as_srt_params", shared.OptionInfo(['Noise Schedule'], "Choose which animation params are to be saved to the .srt file (Frame # and Seed will always be saved):", ui_components.DropdownMulti, lambda: {"interactive": True, "choices": srt_ui_params}, section=section)) 
        
from deforum_helpers.ui_right import on_ui_tabs
script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
