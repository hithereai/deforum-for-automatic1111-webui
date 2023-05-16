


def DeforumAnimPrompts():
    return r"""{
    "0": "tiny cute swamp bunny, highly detailed, intricate, ultra hd, sharp photo, crepuscular rays, in focus, by tomasz alen kopera",
    "30": "anthropomorphic clean cat, surrounded by fractals, epic angle and pose, symmetrical, 3d, depth of field, ruan jia and fenghua zhong",
    "60": "a beautiful coconut --neg photo, realistic",
    "90": "a beautiful durian, trending on Artstation"
}
    """
    

# Guided images defaults    
def keyframeExamples():
    return '''{
    "0": "https://deforum.github.io/a1/Gi1.png",
    "max_f/4-5": "https://deforum.github.io/a1/Gi2.png",
    "max_f/2-10": "https://deforum.github.io/a1/Gi3.png",
    "3*max_f/4-15": "https://deforum.github.io/a1/Gi4.jpg",
    "max_f-20": "https://deforum.github.io/a1/Gi1.png"
}'''

def get_hybrid_info_html():
    hybrid_html = """
    <p style="padding-bottom:0">
        <b style="text-shadow: blue -1px -1px;">Hybrid Video Compositing in 2D/3D Mode</b>
        <span style="color:#DDD;font-size:0.7rem;text-shadow: black -1px -1px;margin-left:10px;">
            by <a href="https://github.com/reallybigname">reallybigname</a>
        </span>
    </p>
    <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em;">
        <li>Composite video with previous frame init image in <b>2D or 3D animation_mode</b> <i>(not for Video Input mode)</i></li>
        <li>Uses your <b>Init</b> settings for <b>video_init_path, extract_nth_frame, overwrite_extracted_frames</b></li>
        <li>In Keyframes tab, you can also set <b>color_coherence</b> = '<b>Video Input</b>'</li>
        <li><b>color_coherence_video_every_N_frames</b> lets you only match every N frames</li>
        <li>Color coherence may be used with hybrid composite off, to just use video color.</li>
        <li>Hybrid motion may be used with hybrid composite off, to just use video motion.</li>
    </ul>
    Hybrid Video Schedules
    <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em;">
        <li>The alpha schedule controls overall alpha for video mix, whether using a composite mask or not.</li>
        <li>The <b>hybrid_comp_mask_blend_alpha_schedule</b> only affects the 'Blend' <b>hybrid_comp_mask_type</b>.</li>
        <li>Mask contrast schedule is from 0-255. Normal is 1. Affects all masks.</li>
        <li>Autocontrast low/high cutoff schedules 0-100. Low 0 High 100 is full range. <br>(<i><b>hybrid_comp_mask_auto_contrast</b> must be enabled</i>)</li>
    </ul>
    <a style='color:SteelBlue;' target='_blank' href='https://github.com/deforum-art/deforum-for-automatic1111-webui/wiki/Animation-Settings#hybrid-video-mode-for-2d3d-animations'>Click Here</a> for more info/ a Guide.
    """
    return hybrid_html
