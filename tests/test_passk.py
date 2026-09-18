import copy
import unittest

from groove.passk import compare_passk, summarize_passk


class PassKTest(unittest.TestCase):
    def records(self, counts=(0, 1, 2, 8)):
        return [dict(question_id=str(q), sample_index=i, accuracy=float(i < count), category='attributes',
                     question=f'Question {q}', ground_truth='A', image_sha256=f'image{q}', prompt_sha256=f'prompt{q}')
                for q, count in enumerate(counts) for i in range(8)]

    def test_pass8_counts_questions_once_and_pass1_averages_all_samples(self):
        result = summarize_passk(self.records(), expected_question_ids=['0', '1', '2', '3'])['overall']
        self.assertEqual(result['samples'], 32)
        self.assertEqual(result['passed_questions'], 3)
        self.assertEqual(result['pass_at_k']['8'], 0.75)
        self.assertEqual(result['pass_at_k']['1'], 11 / 32)
        self.assertAlmostEqual(result['pass_at_k']['4'], (0 + .5 + (1 - 15 / 70) + 1) / 4)

    def test_partial_duplicate_nonbinary_or_misaligned_groups_are_rejected(self):
        records = self.records()
        bad = [records[:-1], records + [records[0]]]
        wrong_index = copy.deepcopy(records); wrong_index[0]['sample_index'] = 1; bad.append(wrong_index)
        nonbinary = copy.deepcopy(records); nonbinary[0]['accuracy'] = .8; bad.append(nonbinary)
        changed = copy.deepcopy(records); changed[0]['prompt_sha256'] = 'different'; bad.append(changed)
        for rows in bad:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                summarize_passk(rows)
        with self.assertRaises(ValueError):
            summarize_passk(records, expected_question_ids=['0', '1'])

    def test_pairing_reports_both_gains_and_losses_and_rejects_input_differences(self):
        base = summarize_passk(self.records((1, 1, 0, 0)))
        best = summarize_passk(self.records((0, 1, 1, 0)))
        result = compare_passk(base, best)
        self.assertEqual(result['difference_percentage_points'], 0)
        self.assertEqual(result['gained_question_ids'], ['2'])
        self.assertEqual(result['lost_question_ids'], ['0'])
        best['per_question'][0]['image_sha256'] = 'different'
        with self.assertRaises(ValueError):
            compare_passk(base, best)


if __name__ == '__main__':
    unittest.main()
