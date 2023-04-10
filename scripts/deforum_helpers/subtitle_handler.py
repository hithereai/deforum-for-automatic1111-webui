from decimal import Decimal, getcontext

def time_to_srt_format(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds, milliseconds = divmod(remainder, 1)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{int(milliseconds * 1000):03}"

def init_srt_file(filename, fps, precision=20):
    with open(filename, "w") as f:
        pass
    getcontext().prec = precision
    frame_duration = Decimal(1) / Decimal(fps)
    return frame_duration

def write_frame_subtitle(filename, frame_number, frame_duration, text):
    frame_start_time = Decimal(frame_number) * frame_duration
    frame_end_time = (Decimal(frame_number) + Decimal(1)) * frame_duration

    with open(filename, "a") as f:
        f.write(f"{frame_number + 1}\n")
        f.write(f"{time_to_srt_format(frame_start_time)} --> {time_to_srt_format(frame_end_time)}\n")
        f.write(f"{text}\n\n")
        
def format_animation_params(keys, frame_idx):
    params_string = ""
    for key in keys.__dict__:
        if key.endswith("_series"):
            params_string += f"{key.replace('_series','').capitalize()}: {keys.__dict__[key][frame_idx]}; "
    params_string = params_string.rstrip("; ")  # Remove trailing semicolon and whitespace
    return params_string

param_dict = {
    "angle": {"backend": "angle_series", "user": "Angle", "print": "Angle"},
    "transform_center_x": {"backend": "transform_center_x_series", "user": "Transform Center X", "print": "Tr.C.X"},
    "transform_center_y": {"backend": "transform_center_y_series", "user": "Transform Center Y", "print": "Tr.C.Y"},
    "zoom": {"backend": "zoom_series", "user": "Zoom", "print": "Zoom"},
    "translation_x": {"backend": "translation_x_series", "user": "Translation X", "print": "TrX"},
    "translation_y": {"backend": "translation_y_series", "user": "Translation Y", "print": "TrY"},
    "translation_z": {"backend": "translation_z_series", "user": "Translation Z", "print": "TrZ"},
    "rotation_3d_x": {"backend": "rotation_3d_x_series", "user": "Rotation 3D X", "print": "RotX"},
    "rotation_3d_y": {"backend": "rotation_3d_y_series", "user": "Rotation 3D Y", "print": "RotY"},
    "rotation_3d_z": {"backend": "rotation_3d_z_series", "user": "Rotation 3D Z", "print": "RotZ"},
    "perspective_flip_theta": {"backend": "perspective_flip_theta_series", "user": "Perspective Flip Theta", "print": "PerFlT"},
    "perspective_flip_phi": {"backend": "perspective_flip_phi_series", "user": "Perspective Flip Phi", "print": "PerFlP"},
    "perspective_flip_gamma": {"backend": "perspective_flip_gamma_series", "user": "Perspective Flip Gamma", "print": "PerFlG"},
    "perspective_flip_fv": {"backend": "perspective_flip_fv_series", "user": "Perspective Flip FV", "print": "PerFlFV"},
    "noise_schedule": {"backend": "noise_schedule_series", "user": "Noise Schedule", "print": "Noise"},
    "strength_schedule": {"backend": "strength_schedule_series", "user": "Strength Schedule", "print": "StrSch"},
    "contrast_schedule": {"backend": "contrast_schedule_series", "user": "Contrast Schedule", "print": "CtrstSch"},
    "cfg_scale_schedule": {"backend": "cfg_scale_schedule_series", "user": "CFG Scale Schedule", "print": "CFGSch"},
    "pix2pix_img_cfg_scale_schedule": {"backend": "pix2pix_img_cfg_scale_series", "user": "Pix2pix Img CFG Scale Schedule", "print": "P2PCfgSch"},
    "subseed_schedule": {"backend": "subseed_schedule_series", "user": "Subseed Schedule", "print": "SubSSch"},
    "subseed_strength_schedule": {"backend": "subseed_strength_schedule_series", "user": "Subseed Strength Schedule", "print": "SubSStrSch"},
    "checkpoint_schedule": {"backend": "checkpoint_schedule_series", "user": "Checkpoint Schedule", "print": "CkptSch"},
    "steps_schedule": {"backend": "steps_schedule_series", "user": "Steps Schedule", "print": "StepsSch"},
    "seed_schedule": {"backend": "seed_schedule_series", "user": "Seed Schedule", "print": "SeedSch"},
    "sampler_schedule": {"backend": "sampler_schedule_series", "user": "Sampler Schedule", "print": "SamplerSchedule"},
    "clipskip_schedule": {"backend": "clipskip_schedule_series", "user": "Clipskip Schedule", "print": "ClipskipSchedule"},
    "noise_multiplier_schedule": {"backend": "noise_multiplier_schedule_series", "user": "Noise Multiplier Schedule", "print": "NoiseMultiplierSchedule"},
    "mask_schedule": {"backend": "mask_schedule_series", "user": "Mask Schedule", "print": "MaskSchedule"},
    "noise_mask_schedule": {"backend": "noise_mask_schedule_series", "user": "Noise Mask Schedule", "print": "NoiseMaskSchedule"},
    "kernel_schedule": {"backend": "kernel_schedule_series", "user": "Kernel Schedule", "print": "KernelSchedule"},
    "sigma_schedule": {"backend": "sigma_schedule_series", "user": "Sigma Schedule", "print": "SigmaSchedule"},
    "amount_schedule": {"backend": "amount_schedule_series", "user": "Amount Schedule", "print": "AmountSchedule"},
    "threshold_schedule": {"backend": "threshold_schedule_series", "user": "Threshold Schedule", "print": "ThresholdSchedule"},
    "aspect_ratio_schedule": {"backend": "aspect_ratio_series", "user": "Aspect Ratio Schedule", "print": "AspectRatioSchedule"},
    "fov_schedule": {"backend": "fov_series", "user": "Field of View Schedule", "print": "FieldOfViewSchedule"},
    "near_schedule": {"backend": "near_series", "user": "Near Schedule", "print": "NearSchedule"},
    "cadence_flow_factor_schedule": {"backend": "cadence_flow_factor_schedule_series", "user": "Cadence Flow Factor Schedule", "print": "CadenceFlowFactorSchedule"},
    "redo_flow_factor_schedule": {"backend": "redo_flow_factor_schedule_series", "user": "Redo Flow Factor Schedule", "print": "RedoFlowFactorSchedule"},
    "far_schedule": {"backend": "far_series", "user": "Far Schedule", "print": "FarSchedule"},
    "hybrid_comp_alpha_schedule": {"backend": "hybrid_comp_alpha_schedule_series", "user": "Hybrid Comp Alpha Schedule", "print": "HybridCompAlphaSchedule"},
    "hybrid_comp_mask_blend_alpha_schedule": {"backend": "hybrid_comp_mask_blend_alpha_schedule_series", "user": "Hybrid Comp Mask Blend Alpha Schedule", "print": "HybridCompMaskBlendAlphaSchedule"},
    "hybrid_comp_mask_contrast_schedule": {"backend": "hybrid_comp_mask_contrast_schedule_series", "user": "Hybrid Comp Mask Contrast Schedule", "print": "HybridCompMaskContrastSchedule"},
    "hybrid_comp_mask_auto_contrast_cutoff_high_schedule": {"backend": "hybrid_comp_mask_auto_contrast_cutoff_high_schedule_series", "user": "Hybrid Comp Mask Auto Contrast Cutoff High Schedule", "print": "HybridCompMaskAutoContrastCutoffHighSchedule"},
    "hybrid_comp_mask_auto_contrast_cutoff_low_schedule": {"backend": "hybrid_comp_mask_auto_contrast_cutoff_low_schedule_series", "user": "Hybrid Comp Mask Auto Contrast Cutoff Low Schedule", "print": "HybridCompMaskAutoContrastCutoffLowSchedule"},
    "hybrid_flow_factor_schedule": {"backend": "hybrid_flow_factor_schedule_series", "user": "Hybrid Flow Factor Schedule", "print": "HybridFlowFactorSchedule"},
}


