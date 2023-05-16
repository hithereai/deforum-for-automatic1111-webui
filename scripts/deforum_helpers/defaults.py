


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
    return """
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
    
def get_composable_masks_info_html():
    return """
        <ul style="list-style-type:circle; margin-left:0.75em; margin-bottom:0.2em">
        <li>To enable, check use_mask in the Init tab</li>
        <li>Supports boolean operations: (! - negation, & - and, | - or, ^ - xor, \ - difference, () - nested operations)</li>
        <li>default variables: in \{\}, like \{init_mask\}, \{video_mask\}, \{everywhere\}</li>
        <li>masks from files: in [], like [mask1.png]</li>
        <li>description-based: <i>word masks</i> in &lt;&gt;, like &lt;apple&gt;, &lt;hair&gt</li>
        </ul>
        """
def get_parseq_info_html():
    return """
        <p>Use a <a style='color:SteelBlue;' target='_blank' href='https://sd-parseq.web.app/deforum'>Parseq</a> manifest for your animation (leave blank to ignore).</p>
        <p style="margin-top:1em; margin-bottom:1em;">
            Fields managed in your Parseq manifest override the values and schedules set in other parts of this UI. You can select which values to override by using the "Managed Fields" section in Parseq.
        </p>
        """
def get_prompts_info_html():
    return """
        <ul style="list-style-type:circle; margin-left:0.75em; margin-bottom:0.2em">
        <li>Please always keep values in math functions above 0.</li>
        <li>There is *no* Batch mode like in vanilla deforum. Please Use the txt2img tab for that.</li>
        <li>For negative prompts, please write your positive prompt, then --neg ugly, text, assymetric, or any other negative tokens of your choice. OR:</li>
        <li>Use the negative_prompts field to automatically append all words as a negative prompt. *Don't* add --neg in the negative_prompts field!</li>
        <li>Prompts are stored in JSON format. If you've got an error, check it in a <a style="color:SteelBlue" href="https://odu.github.io/slingjsonlint/">JSON Validator</a></li>
        </ul>
        """

def get_guided_imgs_info_html():
    return """        
            <p>You can use this as a guided image tool or as a looper depending on your settings in the keyframe images field. 
            Set the keyframes and the images that you want to show up. 
            Note: the number of frames between each keyframe should be greater than the tweening frames.</p>
            
            <p>In later versions, this should also be in the strength schedule, but for now, you need to set it.</p>
            
            <p>Prerequisites and Important Info:</p>
            <ul style="list-style-type:circle; margin-left:2em; margin-bottom:0em">
                <li>This mode works ONLY with 2D/3D animation modes. Interpolation and Video Input modes aren't supported.</li>
                <li>Init tab's strength slider should be greater than 0. Recommended value (.65 - .80).</li>
                <li>'seed_behavior' will be forcibly set to 'schedule'.</li>
            </ul>
            
            <p>Looping recommendations:</p>
            <ul style="list-style-type:circle; margin-left:2em; margin-bottom:0em">
                <li>seed_schedule should start and end on the same seed.<br />
                Example: seed_schedule could use 0:(5), 1:(-1), 219:(-1), 220:(5)</li>
                <li>The 1st and last keyframe images should match.</li>
                <li>Set your total number of keyframes to be 21 more than the last inserted keyframe image.<br />
                Example: Default args should use 221 as the total keyframes.</li>
                <li>Prompts are stored in JSON format. If you've got an error, check it in the validator, 
                <a style="color:SteelBlue" href="https://odu.github.io/slingjsonlint/">like here</a></li>
            </ul>
            
            <p>The Guided images mode exposes the following variables for the prompts and the schedules:</p>
            <ul style="list-style-type:circle; margin-left:2em; margin-bottom:0em">
                <li><b>s</b> is the <i>initial</i> seed for the whole video generation.</li>
                <li><b>max_f</b> is the length of the video, in frames.<br />
                Example: seed_schedule could use 0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)</li>
                <li><b>t</b> is the current frame number.<br />
                Example: strength_schedule could use 0:(0.25 * cos((72 / 60 * 3.141 * (t + 0) / 30))**13 + 0.7) to make alternating changes each 30 frames</li>
            </ul>
          """
    
    return html_string

def get_gradio_html(section_name):
    if section_name.lower() == 'hybrid_video':
        return get_hybrid_info_html()
    elif section_name.lower() == 'composable_masks':
        return get_composable_masks_info_html()
    elif section_name.lower() == 'parseq':
        return get_parseq_info_html()
    elif section_name.lower() == 'prompts':
        return get_prompts_info_html()
    elif section_name.lower() == 'guided_imgs':
        return get_guided_imgs_info_html()
    else:
        return None
        