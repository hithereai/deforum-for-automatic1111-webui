##
# From /scripts directory, run like: python -m unittest deforum_helpers.parseq_adapter_test
##

import unittest
from .parseq_adapter import ParseqAdapter 
from .animation_key_frames import DeformAnimKeys, LooperAnimKeys, ControlNetKeys
from unittest.mock import patch
from unittest.mock import MagicMock, PropertyMock
from types import SimpleNamespace

DEFAULT_ARGS = SimpleNamespace(anim_args = SimpleNamespace(max_frames=2),
                               video_args = SimpleNamespace(fps=30),
                               args = SimpleNamespace(seed=-1),
                               loop_args = SimpleNamespace(),
                               controlnet_args = SimpleNamespace())


def buildParseqAdapter(parseq_use_deltas, parseq_manifest, setup_args=DEFAULT_ARGS):
    return ParseqAdapter(SimpleNamespace(parseq_use_deltas=parseq_use_deltas, parseq_manifest=parseq_manifest),
                         setup_args.anim_args, setup_args.video_args, setup_args.controlnet_args)

class TestParseqAnimKeys(unittest.TestCase):

    @patch('deforum_helpers.parseq_adapter.DeformAnimKeys')
    @patch('deforum_helpers.parseq_adapter.ControlNetKeys')
    def test_withprompt(self, mock_deformanimkeys, mock_controlnetkeys):
        parseq_adapter = buildParseqAdapter(parseq_use_deltas=True, parseq_manifest=""" 
            {                
                "options": {
                    "output_fps": 30
                },
                "rendered_frames": [
                    {
                        "frame": 0,
                        "deforum_prompt": "blah"
                    },
                    {
                        "frame": 1,
                        "deforum_prompt": "blah"
                    }
                ]
            }
            """)
        self.assertTrue(parseq_adapter.manages_prompts())


    @patch('deforum_helpers.parseq_adapter.DeformAnimKeys')
    @patch('deforum_helpers.parseq_adapter.ControlNetKeys')
    def test_withoutprompt(self, mock_deformanimkeys, mock_controlnetkeys):
        parseq_adapter = buildParseqAdapter(parseq_use_deltas=True, parseq_manifest=""" 
            {                
                "options": {
                    "output_fps": 30
                },
                "rendered_frames": [
                    {
                        "frame": 0
                    },
                    {
                        "frame": 1
                    }
                ]
            }
            """)
        self.assertFalse(parseq_adapter.manages_prompts())

    @patch('deforum_helpers.parseq_adapter.DeformAnimKeys')
    @patch('deforum_helpers.parseq_adapter.ControlNetKeys')
    def test_withseed(self, mock_deformanimkeys, mock_controlnetkeys):
        parseq_adapter = buildParseqAdapter(parseq_use_deltas=True, parseq_manifest=""" 
            {                
                "options": {
                    "output_fps": 30
                },
                "rendered_frames": [
                    {
                        "frame": 0,
                        "seed": 1
                    },
                    {
                        "frame": 1,
                        "seed": 2
                    }
                ]
            }
            """)
        self.assertTrue(parseq_adapter.manages_seed())


    @patch('deforum_helpers.parseq_adapter.DeformAnimKeys')
    @patch('deforum_helpers.parseq_adapter.ControlNetKeys')
    def test_withoutseed(self, mock_deformanimkeys, mock_controlnetkeys):
        parseq_adapter = buildParseqAdapter(parseq_use_deltas=True, parseq_manifest=""" 
            {                
                "options": {
                    "output_fps": 30
                },
                "rendered_frames": [
                    {
                        "frame": 0
                    },
                    {
                        "frame": 1
                    }
                ]
            }
            """)
        self.assertFalse(parseq_adapter.manages_seed())


    @patch('deforum_helpers.parseq_adapter.DeformAnimKeys')
    @patch('deforum_helpers.parseq_adapter.ControlNetKeys')
    def test_usedelta(self, mock_deformanimkeys, mock_controlnetkeys):
        parseq_adapter = buildParseqAdapter(parseq_use_deltas=True, parseq_manifest=""" 
            {                
                "options": {
                    "output_fps": 30
                },
                "rendered_frames": [
                    {
                        "frame": 0,
                        "angle": 90,
                        "angle_delta": 90
                    },
                    {
                        "frame": 1,
                        "angle": 180,
                        "angle_delta": 90
                    }
                ]
            }
            """)
        self.assertEqual(parseq_adapter.anim_keys.angle_series[1], 90)

    @patch('deforum_helpers.parseq_adapter.DeformAnimKeys')
    @patch('deforum_helpers.parseq_adapter.ControlNetKeys')
    def test_usenondelta(self, mock_deformanimkeys, mock_controlnetkeys):
        parseq_adapter = buildParseqAdapter(parseq_use_deltas=False, parseq_manifest=""" 
            {                
                "options": {
                    "output_fps": 30
                },
                "rendered_frames": [
                    {
                        "frame": 0,
                        "angle": 90,
                        "angle_delta": 90
                    },
                    {
                        "frame": 1,
                        "angle": 180,
                        "angle_delta": 90
                    }
                ]
            }
            """)
        self.assertEqual(parseq_adapter.anim_keys.angle_series[1], 180)

    @patch('deforum_helpers.parseq_adapter.DeformAnimKeys')
    @patch('deforum_helpers.parseq_adapter.ControlNetKeys')
    def test_fallbackonundefined(self, mock_deformanimkeys, mock_controlnetkeys):
        parseq_adapter = buildParseqAdapter(parseq_use_deltas=False, parseq_manifest=""" 
            {                
                "options": {
                    "output_fps": 30
                },
                "rendered_frames": [
                    {
                        "frame": 0
                    },
                    {
                        "frame": 1
                    }
                ]
            }
            """)
        #TODO - this is a hacky check to make sure we're falling back to the mock.
        #There must be a better way to inject an expected value via patch and check for that...
        self.assertRegex(str(parseq_adapter.anim_keys.angle_series[0]), r'MagicMock')

    @patch('deforum_helpers.parseq_adapter.DeformAnimKeys')
    @patch('deforum_helpers.parseq_adapter.ControlNetKeys')
    def test_cn(self, mock_deformanimkeys, mock_controlnetkeys):
        parseq_adapter = buildParseqAdapter(parseq_use_deltas=False, parseq_manifest=""" 
            {                
                "options": {
                    "output_fps": 30
                },
                "rendered_frames": [
                    {
                        "frame": 0,
                        "cn_1_weight": 1
                    },
                    {
                        "frame": 1,
                        "cn_1_weight": 1
                    }
                ]
            }
            """)
        self.assertEqual(parseq_adapter.cn_keys.cn_1_weight_schedule_series[0], 1)

    @patch('deforum_helpers.parseq_adapter.DeformAnimKeys')
    @patch('deforum_helpers.parseq_adapter.ControlNetKeys')
    def test_cn_fallback(self, mock_deformanimkeys, mock_controlnetkeys):
        parseq_adapter = buildParseqAdapter(parseq_use_deltas=False, parseq_manifest=""" 
            {                
                "options": {
                    "output_fps": 30
                },
                "rendered_frames": [
                    {
                        "frame": 0
                    },
                    {
                        "frame": 1
                    }
                ]
            }
            """)
        #TODO - this is a hacky check to make sure we're falling back to the mock.
        #There must be a better way to inject an expected value via patch and check for that...
        self.assertRegex(str(parseq_adapter.cn_keys.cn_1_weight_schedule_series[0]), r'MagicMock')           
        
if __name__ == '__main__':
    unittest.main()