import unittest
from unittest.mock import patch

import workflow
from agents import AgentError


class WorkflowTests(unittest.TestCase):
    def test_run_rejects_blank_input(self):
        with self.assertRaises(AgentError):
            workflow.run("  ")

    def test_rejected_verdict_is_revised_then_reported(self):
        reviews = iter([
            {"approved": False, "feedback": "Check the evidence again."},
            {"approved": True, "feedback": "Supported."},
        ])
        judge_calls = []

        def judge(claim, evidence, feedback=None):
            judge_calls.append(feedback)
            return {
                "verdict": "True",
                "confidence": 90,
                "reasoning": "Supported by source [1].",
            }

        with (
            patch.object(workflow, "extract_claims", return_value=["The Earth orbits the Sun."]),
            patch.object(workflow, "retrieve_evidence", return_value=("Earth orbit", [{"title": "Source"}])),
            patch.object(workflow, "judge_claim", side_effect=judge),
            patch.object(workflow, "review_verdict", side_effect=lambda *_: next(reviews)),
        ):
            result = workflow.run("The Earth orbits the Sun.")

        self.assertEqual(result["overall"], "True")
        self.assertEqual(len(judge_calls), 2)
        self.assertIsNone(judge_calls[0])
        self.assertEqual(judge_calls[1], "Check the evidence again.")
        self.assertEqual(result["retries"], 2)


if __name__ == "__main__":
    unittest.main()
