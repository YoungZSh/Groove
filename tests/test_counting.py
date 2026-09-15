from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from PIL import Image

from groove.analyzer import OpenAIAnalyzerConfig, OpenAICompatibleAnalyzer
from groove.analyzer_tools import AnalyzerVisionToolConfig, AnalyzerVisionToolRegistry
from groove.counting import CountingProfile, DEFAULT_COUNTING_PROFILE, counting_region, select_counting_instances
from groove.schemas import FocusProgram, InstanceBox


class CountingTest(unittest.TestCase):
    def test_default_keeps_overlaps_and_explicit_nms_is_separate(self):
        instances = [
            InstanceBox(bbox=(0, 0, 10, 10), score=.9),
            InstanceBox(bbox=(1, 1, 10, 10), score=.8),
            InstanceBox(bbox=(8, 0, 18, 10), score=.7),
            InstanceBox(bbox=(30, 0, 40, 10), score=.2),
        ]
        self.assertEqual(select_counting_instances(instances), [0, 1, 2])
        self.assertEqual(select_counting_instances(instances, CountingProfile(nms_iou_threshold=.5)), [0, 2])
        self.assertIsNone(DEFAULT_COUNTING_PROFILE.nms_iou_threshold)
        self.assertEqual(DEFAULT_COUNTING_PROFILE.box_threshold, .35)
        self.assertEqual(DEFAULT_COUNTING_PROFILE.text_threshold, .25)

    def test_counting_region_validation(self):
        self.assertEqual(counting_region(None, (100, 80)), (0, 0, 100, 80))
        self.assertEqual(counting_region([1.1, 2.2, 99.8, 40.1], (100, 80)), (1, 2, 100, 41))
        for region in [[20, 20, 10, 30], [0, 0, float('nan'), 20], [110, 0, 120, 10],
                       [-1, 2, 120, 40], [148, 228, 866, 781]]:
            with self.subTest(region=region), self.assertRaises(ValueError):
                counting_region(region, (100, 80))

    def test_count_tool_crops_before_detection_and_returns_original_coordinates(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'original.png'
            Image.new('RGB', (100, 80), (20, 40, 60)).save(path)
            registry = AnalyzerVisionToolRegistry(AnalyzerVisionToolConfig(
                grounding_url='http://dino.invalid', ocr_url='http://ocr.invalid', enable_counting=True,
                enable_instance_boxes=True))
            self.assertEqual([s['function']['name'] for s in registry.schemas],
                             ['ground_image', 'read_text', 'count_objects'])
            before = path.read_bytes()
            def remote(_url, crop_path, payload):
                with Image.open(crop_path) as crop:
                    self.assertEqual(crop.size, (60, 40))
                self.assertEqual(payload['box_threshold'], .35)
                self.assertEqual(payload['text_threshold'], .25)
                self.assertTrue(payload['return_all'])
                return {'image_size': [60, 40], 'instances': [
                    {'bbox': [1, 2, 11, 12], 'score': .8},
                    {'bbox': [2, 3, 11, 12], 'score': .7},
                    {'bbox': [30, 10, 45, 25], 'score': .6},
                ]}
            with patch.object(registry, '_remote_call', side_effect=remote):
                result = registry.execute(path, 'count_objects', {'target': 'sheep', 'region': [20, 10, 80, 50]})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(result['estimated_count'], 3)
            self.assertEqual(result['raw_candidate_count'], 3)
            self.assertEqual(len(result['candidates_before_filtering']), 3)
            self.assertEqual(result['candidates_before_filtering'][1]['bbox'], [22, 13, 31, 22])
            self.assertEqual(result['image_size'], [100, 80])
            self.assertEqual(result['instances'][0]['bbox'], [21, 12, 31, 22])
            self.assertEqual(result['instances'][1]['bbox'], [22, 13, 31, 22])
            self.assertEqual(result['instances'][2]['bbox'], [50, 20, 65, 35])
            self.assertEqual(result['kept_candidate_indices'], [0, 1, 2])
            with self.assertRaisesRegex(ValueError, 'filtering is fixed'):
                registry.execute(path, 'count_objects', {'target': 'sheep', 'box_threshold': .1})
            analyzer = OpenAICompatibleAnalyzer(OpenAIAnalyzerConfig(base_url='http://unused', api_key='unused'))
            analyzer.last_tool_trace = [{'name': 'count_objects', 'candidate_id': 'candidate_1', 'result': result}]
            focus = FocusProgram(group_summary='Candidate sheep.', visible_focus_instruction='Inspect the boxes.',
                                 grounding_queries=['sheep'], selected_candidate_ids=['candidate_1'])
            selected = analyzer._attach_tool_regions(focus, path)
            self.assertEqual(selected.tool_regions[0].kind, 'instance_boxes')
            self.assertEqual(len(selected.tool_regions[0].instances), 3)

    def test_occluded_sheep_boxes_are_not_suppressed_by_default(self):
        instances = [
            InstanceBox(bbox=(177.845, 164.167, 276.557, 220.709), score=.360274),
            InstanceBox(bbox=(173.135, 157.018, 271.147, 210.213), score=.358544),
        ]
        self.assertEqual(select_counting_instances(instances), [0, 1])
        self.assertEqual(select_counting_instances(instances, CountingProfile(nms_iou_threshold=.5)), [0])

    def test_count_tool_rejects_legacy_remote_and_is_opt_in(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'image.png'
            Image.new('RGB', (40, 40)).save(path)
            registry = AnalyzerVisionToolRegistry(AnalyzerVisionToolConfig(
                grounding_url='http://dino.invalid', ocr_url='http://ocr.invalid', enable_counting=True))
            with patch.object(registry, '_remote_call', return_value={'found': True, 'bbox': [0, 0, 10, 10]}):
                with self.assertRaisesRegex(ValueError, 'multi-instance support'):
                    registry.count_objects(path, 'sheep')
            disabled = AnalyzerVisionToolRegistry(AnalyzerVisionToolConfig(
                grounding_url='http://dino.invalid', ocr_url='http://ocr.invalid'))
            with self.assertRaisesRegex(ValueError, 'not enabled'):
                disabled.count_objects(path, 'sheep')


if __name__ == '__main__':
    unittest.main()
