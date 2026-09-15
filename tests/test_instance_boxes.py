from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from groove.analyzer import OpenAIAnalyzerConfig, OpenAICompatibleAnalyzer, StaticAnalyzer, _tool_feedback_message
from groove.analyzer_tools import AnalyzerVisionToolConfig, AnalyzerVisionToolRegistry, GROUND_INSTANCES_TOOL_SCHEMA
from groove.evidence import EvidenceBuilderConfig, TeacherEvidenceBuilder, teacher_payload
from groove.grounding import GroundingDinoGrounder, GroundingDinoConfig, StaticGrounder
from groove.instance_boxes import normalize_instance_boxes, render_instance_boxes
from groove.schemas import FocusProgram, GroupRollout, InstanceBox, Rollout, ToolRegion


class InstanceBoxTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image_path = self.root / 'original.png'
        Image.new('RGB', (200, 100), (20, 40, 60)).save(self.image_path)
        self.instances = [{'bbox': [10 + i * 30, 20, 25 + i * 30, 45], 'score': .8}
                          for i in range(5)]

    def registry(self, enabled=True):
        return AnalyzerVisionToolRegistry(AnalyzerVisionToolConfig(
            grounding_url='http://dino.invalid', ocr_url='http://ocr.invalid',
            enable_instance_boxes=enabled))

    def result(self):
        return {'query': 'sheep', 'image_size': [200, 100], 'found': True,
                'instances': deepcopy(self.instances), 'candidate_id': 'candidate_1'}

    def test_overlay_preserves_scene_and_native_size(self):
        original = Image.open(self.image_path).convert('RGB')
        before = original.tobytes()
        annotated = render_instance_boxes(original, normalize_instance_boxes(self.instances, original.size))
        self.assertEqual(annotated.size, original.size)
        self.assertEqual(original.tobytes(), before)
        self.assertEqual(annotated.getpixel((10, 20)), (255, 0, 0))
        self.assertEqual(annotated.getpixel((15, 30)), (20, 40, 60))  # No fill or count text.
        self.assertEqual(annotated.getpixel((0, 0)), (20, 40, 60))

    def test_boxes_are_clipped_and_duplicates_remain_auditable(self):
        values = [{'bbox': [-2, 2, 201, 30], 'score': .5}] * 2
        boxes = normalize_instance_boxes(values, (200, 100))
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0].bbox, (0, 2, 200, 30))
        for bbox in [[10, 10, 2, 20], [0, 0, float('nan'), 10], [201, 0, 220, 20]]:
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                normalize_instance_boxes([{'bbox': bbox, 'score': .8}], (200, 100))
        with self.assertRaises(ValueError):
            render_instance_boxes(Image.new('RGB', (20, 20)), [])
        tiny = normalize_instance_boxes([{'bbox': [19.8, 19.8, 20, 20], 'score': .7}], (20, 20))
        self.assertEqual(render_instance_boxes(Image.new('RGB', (20, 20)), tiny).size, (20, 20))

    def test_remote_interface_requires_multi_instance_results(self):
        registry = self.registry()
        legacy = {'found': True, 'bbox': [10, 20, 25, 45], 'score': .8, 'image_size': [200, 100]}
        with patch.object(registry, '_remote_call', return_value=legacy):
            with self.assertRaisesRegex(ValueError, 'multi-instance support'):
                registry.ground_instances(self.image_path, 'sheep')
        with patch.object(registry, '_remote_call', return_value=self.result()) as remote:
            output = registry.ground_instances(self.image_path, 'sheep')
        self.assertEqual(len(output['instances']), 5)
        self.assertTrue(remote.call_args.args[2]['return_all'])
        self.assertNotIn('bbox', output)
        self.assertNotIn('count', output)

    def test_tool_is_opt_in_and_empty_or_wrong_size_results_are_not_evidence(self):
        disabled = self.registry(False)
        self.assertNotIn('ground_instances', [s['function']['name'] for s in disabled.schemas])
        with self.assertRaisesRegex(ValueError, 'not enabled'):
            disabled.execute(self.image_path, 'ground_instances', {'query': 'sheep'})
        registry = self.registry()
        self.assertIn('ground_instances', [s['function']['name'] for s in registry.schemas])
        with patch.object(registry, '_remote_call', return_value={**self.result(), 'instances': [], 'found': False}):
            self.assertFalse(registry.ground_instances(self.image_path, 'sheep')['found'])
        with patch.object(registry, '_remote_call', return_value={**self.result(), 'image_size': [100, 100]}):
            with self.assertRaisesRegex(ValueError, 'image size'):
                registry.ground_instances(self.image_path, 'sheep')

    def test_single_object_detector_keeps_highest_score_after_refactor(self):
        grounder = GroundingDinoGrounder(GroundingDinoConfig(device='cpu'))
        values = [((1., 2., 3., 4.), .4), ((5., 6., 7., 8.), .9)]
        with patch.object(grounder, '_detect_all', return_value=values):
            self.assertEqual(grounder._detect_one(Image.new('RGB', (20, 20)), 'sheep'), values[1])
        with patch.object(grounder, '_detect_all', return_value=[]):
            self.assertIsNone(grounder._detect_one(Image.new('RGB', (20, 20)), 'sheep'))

    def test_analyzer_sees_overlay_and_selects_all_boxes_as_one_candidate(self):
        result = self.result()
        registry = SimpleNamespace(schemas=[GROUND_INSTANCES_TOOL_SCHEMA], execute=lambda *_args: deepcopy(result))
        response_focus = {'group_summary': 'Inspect the candidate animals.',
                          'visible_focus_instruction': 'Verify the candidate sheep against the original image.',
                          'grounding_queries': ['sheep'], 'selected_candidate_ids': ['candidate_1']}
        bodies = []
        class FakeAnalyzer(OpenAICompatibleAnalyzer):
            def _request(self, body):
                bodies.append(deepcopy(body))
                if len(bodies) == 1:
                    return {'choices': [{'message': {'content': '', 'tool_calls': [{
                        'id': 'call1', 'type': 'function',
                        'function': {'name': 'ground_instances', 'arguments': '{"query":"sheep"}'}}]}}]}
                return {'choices': [{'message': {'content': json.dumps(response_focus)}}]}
        analyzer = FakeAnalyzer(OpenAIAnalyzerConfig(base_url='http://unused', api_key='unused', use_vision_tools=True))
        group = GroupRollout(uid='test', question='How many sheep?', image_path=self.image_path,
                             rollouts=[Rollout(rollout_id=0, completion='<answer>5</answer>', predicted_label=None, is_correct=True)])
        with patch('groove.analyzer_tools.AnalyzerVisionToolRegistry', return_value=registry):
            focus = analyzer.analyze(group)
        self.assertIn('For counting, use ground_instances', bodies[0]['messages'][0]['content'])
        self.assertEqual(len(focus.tool_regions), 1)
        self.assertEqual(focus.tool_regions[0].kind, 'instance_boxes')
        self.assertEqual(len(focus.tool_regions[0].instances), 5)
        self.assertTrue(analyzer.last_tool_trace[0]['visual_feedback_attached'])
        feedback = bodies[1]['messages'][-1]
        data_url = feedback['content'][1]['image_url']['url']
        with Image.open(BytesIO(base64.b64decode(data_url.split(',', 1)[1]))) as preview:
            self.assertEqual(preview.size, (200, 100))
            self.assertEqual(preview.getpixel((10, 20)), (255, 0, 0))
        self.assertNotIn('ground_truth', bodies[0]['messages'][1]['content'][1]['text'])

    def test_overlay_builder_keeps_student_untouched_and_teacher_image_order(self):
        region = ToolRegion(query='sheep', expanded_box=(0, 0, 200, 100), score=.8,
                            source='grounding_dino', kind='instance_boxes',
                            instances=[InstanceBox.model_validate(x) for x in self.instances])
        crop = ToolRegion(query='ambiguous detail', expanded_box=(0, 0, 30, 30), score=.7, source='grounding_dino')
        focus = FocusProgram(group_summary='Private evidence.', visible_focus_instruction='Inspect the candidate sheep.',
                             grounding_queries=['sheep'], tool_regions=[region, crop])
        analyzer = StaticAnalyzer(focus)
        analyzer.last_tool_trace = [{'name': 'ground_instances', 'private_note': 'AUDIT_ONLY_SENTINEL'}]
        group = GroupRollout(uid='mixed', question='How many sheep?', image_path=self.image_path,
            rollouts=[Rollout(rollout_id=i, completion='response', predicted_label=None, is_correct=False) for i in range(2)])
        student = [{'role': 'system', 'content': 'Original instruction.'},
                   {'role': 'user', 'content': '<image>How many sheep?'}]
        before = deepcopy(student)
        image_before = self.image_path.read_bytes()
        builder = TeacherEvidenceBuilder(analyzer, StaticGrounder([]), EvidenceBuilderConfig(output_dir=self.root / 'out'))
        evidence = builder.build(group, student_prompt=student)
        self.assertEqual(evidence.status, 'ready')
        self.assertEqual(student, before)
        self.assertEqual(self.image_path.read_bytes(), image_before)
        prompt, images = teacher_payload(evidence, question=group.question)
        self.assertEqual(len(images), 3)
        self.assertEqual(images[0]['path'], str(self.image_path))
        self.assertEqual(images[1]['path'], str(evidence.crops[0].path))
        self.assertEqual([c.kind for c in evidence.crops], ['instance_boxes', 'crop'])
        self.assertEqual(len(evidence.crops[0].instances), 5)
        self.assertEqual(prompt[0], student[0])
        self.assertIn('Candidate instance boxes 1:', prompt[1]['content'])
        self.assertIn('Zoomed visual evidence 2:', prompt[1]['content'])
        self.assertNotIn('AUDIT_ONLY_SENTINEL', str(prompt))
        saved = json.loads((self.root / 'out/mixed/evidence.json').read_text())
        self.assertIn('AUDIT_ONLY_SENTINEL', str(saved['tool_trace']))
        # Cache reload must preserve kinds and rebuild the same Teacher protocol.
        again = builder.build(group, student_prompt=student)
        self.assertEqual(teacher_payload(again, question=group.question), (prompt, images))

    def test_invalid_box_set_falls_back_without_teacher_images(self):
        class BrokenAnalyzer:
            last_tool_trace = [{'name': 'ground_instances', 'result': {'error': 'bad coordinates'}}]
            def analyze(self, _group):
                raise ValueError('invalid instance coordinates')
        group = GroupRollout(uid='broken', question='How many?', image_path=self.image_path,
            rollouts=[Rollout(rollout_id=i, completion='x', predicted_label=None, is_correct=False) for i in range(2)])
        builder = TeacherEvidenceBuilder(BrokenAnalyzer(), StaticGrounder([]), EvidenceBuilderConfig(output_dir=self.root / 'out'))
        evidence = builder.build(group)
        self.assertEqual(evidence.status, 'error')
        self.assertEqual(teacher_payload(evidence, question=group.question)[1], [])
        self.assertEqual(evidence.tool_trace[0]['name'], 'ground_instances')


class RemoteDinoInstancesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'remote_tools/dino_server.py'
        spec = importlib.util.spec_from_file_location('testable_dino_server', path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def run_ground(self, return_all=False, empty=False):
        image = BytesIO()
        Image.new('RGB', (40, 30)).save(image, format='PNG')
        boxes = torch.empty((0, 4)) if empty else torch.tensor([[1., 2., 10., 12.], [20., 3., 30., 15.]])
        scores = torch.empty((0,)) if empty else torch.tensor([.4, .9])
        class Processor:
            def __call__(self, **_kwargs):
                return {'input_ids': SimpleNamespace(cuda=lambda: object())}
            def post_process_grounded_object_detection(self, *_args, **_kwargs):
                return [{'boxes': boxes, 'scores': scores}]
        payload = {'image_base64': base64.b64encode(image.getvalue()).decode(), 'query': 'sheep', 'context_margin': .12}
        if return_all:
            payload['return_all'] = True
        with patch.object(self.module, 'processor', Processor()), patch.object(self.module, 'model', lambda **_kwargs: object()), patch('torch.cuda.empty_cache'):
            return self.module.ground(payload)

    def test_single_and_multi_instance_protocols(self):
        single = self.run_ground()
        self.assertEqual(single['raw_bbox'], [20., 3., 30., 15.])
        self.assertNotIn('instances', single)
        multiple = self.run_ground(True)
        self.assertEqual(len(multiple['instances']), 2)
        self.assertEqual(multiple['instances'][0]['bbox'], [1., 2., 10., 12.])
        self.assertEqual(multiple['instances'][1]['bbox'], [20., 3., 30., 15.])
        self.assertNotIn('bbox', multiple)

    def test_empty_multi_instance_result(self):
        result = self.run_ground(True, empty=True)
        self.assertFalse(result['found'])
        self.assertEqual(result['instances'], [])


if __name__ == '__main__':
    unittest.main()
