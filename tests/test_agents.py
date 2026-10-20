import unittest
from unittest.mock import patch

import agents


class AgentTests(unittest.TestCase):
    def test_parse_json_accepts_fenced_json(self):
        self.assertEqual(agents.parse_json('```json\n{"approved": false}\n```'), {"approved": False})

    def test_judge_marks_malformed_response_unverified(self):
        with patch.object(agents, "ask_llm", return_value="not JSON"):
            result = agents.judge_claim("A claim", [])

        self.assertEqual(result["verdict"], "Unverified")
        self.assertIn("manual review", result["reasoning"])

    def test_reviewer_does_not_approve_non_boolean_response(self):
        with patch.object(agents, "ask_llm", return_value='{"approved": "false"}'):
            result = agents.review_verdict("A claim", [], {
                "verdict": "True",
                "confidence": 80,
                "reasoning": "Reason",
            })

        self.assertFalse(result["approved"])

    def test_extractor_rejects_non_list_response(self):
        with patch.object(agents, "ask_llm", return_value='{"claim": "not a list"}'):
            with self.assertRaises(agents.AgentError):
                agents.extract_claims("A claim")

    def test_web_search_dependency_error_is_not_reported_as_no_results(self):
        with patch.object(agents, "MOCK", False), patch("builtins.__import__", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(agents.AgentError, "web-search package is missing"):
                agents.web_search("A query")


if __name__ == "__main__":
    unittest.main()
