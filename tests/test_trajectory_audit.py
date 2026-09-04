from types import SimpleNamespace
import json

import numpy as np
import torch

from groove.trajectory_audit import write_trajectory_audit


def test_audit_preserves_all_valid_tokens_and_batch_identity(tmp_path):
    # Include a masked hole, EOS, weak credit, and a group with no evidence.
    mask = torch.tensor([[1, 0, 1], [1, 1, 0]])
    student = torch.tensor([[-1., -99., -2.], [-3., -4., -99.]])
    gap = torch.tensor([[0., 50., -1.], [2., 3., 50.]])
    grpo = torch.tensor([[1., 0., 1.], [-1., -1., 0.]])
    opsd = torch.tensor([[0., 0., -1.], [0., 0., 0.]])
    batch = SimpleNamespace(
        batch={"response_mask": mask, "responses": torch.tensor([[10, 99, 2], [20, 2, 99]])},
        non_tensor_batch={
            "uid": np.array(["second-group", "first-group"]),
            "extra_info": [{"question_id": "q2"}, {"question_id": "q1"}],
        },
    )
    count = write_trajectory_audit(
        tmp_path, step=7, batch=batch,
        tokenizer=SimpleNamespace(decode=lambda ids, **_: str(ids)),
        student_log_probs=student, teacher_log_probs=student + gap,
        grpo_advantages=grpo, opsd_advantages=opsd,
        total_advantages=grpo + .01 * opsd, evidence_mask=torch.tensor([1, 0]),
        sequence_rewards=torch.tensor([1., .1]), opsd_coef=.01,
    )
    data = torch.load(tmp_path / "7.rank0.pt", weights_only=True)
    records = [json.loads(line) for line in (tmp_path / "7.jsonl").read_text().splitlines()]
    assert count == 4
    assert data["token_ids"].tolist() == [10, 2, 20, 2]
    assert data["response_positions"].tolist() == [0, 2, 0, 1]
    assert data["sample_ids"].tolist() == [0, 0, 1, 1]
    assert data["opsd_advantages"].tolist() == [0., -1., 0., 0.]
    assert data["evidence_mask"].tolist() == [True, True, False, False]
    assert records[0]["question_id"] == "q2"
    assert records[1]["response_token_ids"] == [20, 2]
    assert torch.allclose(data["total_advantages"], torch.tensor([1., .99, -1., -1.]))
    assert not list(tmp_path.glob("*.tmp"))
