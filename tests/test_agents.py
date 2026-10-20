import unittest
import os
from unittest.mock import patch

import agents


class AgentTests(unittest.TestCase):
    def test_parse_json_accepts_fenced_json(self):
        self.assertEqual(agents.parse_json('```json\n{"approved": false}\n```'), {"approved": False})

    def test_generate_answer_passes_custom_prompt_to_model(self):
        with patch.object(agents, "ask_llm", return_value="A generated response.") as ask_llm:
            answer = agents.generate_answer("  Explain photosynthesis.  ")

        self.assertEqual(answer, "A generated response.")
        self.assertEqual(ask_llm.call_args.args[0], "answer")
        self.assertEqual(ask_llm.call_args.args[2], "Explain photosynthesis.")

    def test_generate_answer_rejects_blank_prompt(self):
        with self.assertRaises(agents.AgentError):
            agents.generate_answer("  ")

    def test_ollama_client_uses_local_default_without_api_key(self):
        with (
            patch.object(agents, "_client", None),
            patch.dict(os.environ, {"LLM_API_KEY": "", "LLM_BASE_URL": ""}),
            patch("openai.OpenAI") as client,
        ):
            agents._get_client()

        client.assert_called_once_with(api_key="ollama", base_url="http://localhost:11434/v1")

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

    def test_extractor_accepts_wrapped_claim_objects(self):
        response = (
            '{"claims": [{"text": "The Great Wall is visible from the Moon.", '
            '"type": "fact"}]}'
        )
        with patch.object(agents, "ask_llm", return_value=response):
            claims = agents.extract_claims("The Great Wall is visible from the Moon.")

        self.assertEqual(claims, ["The Great Wall is visible from the Moon."])

    def test_web_search_dependency_error_is_not_reported_as_no_results(self):
        with patch.object(agents, "MOCK", False), patch("builtins.__import__", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(agents.AgentError, "web-search package is missing"):
                agents.web_search("A query")


if __name__ == "__main__":
    unittest.main()
