def init_deforum():
    import sys, os
    from modules import script_callbacks

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

    from deforum_helpers.ui_right import on_ui_tabs
    script_callbacks.on_ui_tabs(on_ui_tabs)
    from deforum_helpers.ui_settings import on_ui_settings
    script_callbacks.on_ui_settings(on_ui_settings)

init_deforum()