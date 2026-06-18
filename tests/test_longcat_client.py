"""测试LongCat API客户端"""
import pytest
from unittest.mock import patch, MagicMock
from openai import RateLimitError


class TestLongCatClient:
    def test_chat_completion(self):
        from backend.models.longcat_client import chat_completion
        with patch("backend.models.longcat_client.client") as mock_client:
            mock_message = MagicMock()
            mock_message.content = "Hello"
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = mock_message
            mock_client.chat.completions.create.return_value = mock_response
            result = chat_completion(system="test", user="hi")
            assert result.content == "Hello"

    def test_chat_completion_with_tools(self):
        from backend.models.longcat_client import chat_completion
        with patch("backend.models.longcat_client.client") as mock_client:
            mock_message = MagicMock()
            mock_message.content = "tool response"
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = mock_message
            mock_client.chat.completions.create.return_value = mock_response
            tools = [{"type": "function", "function": {"name": "test"}}]
            result = chat_completion(system="test", user="hi", tools=tools)
            assert result.content == "tool response"
            # Verify tools and tool_choice were passed
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["tools"] == tools
            assert call_kwargs["tool_choice"] == "auto"

    def test_chat_json(self):
        from backend.models.longcat_client import chat_json
        with patch("backend.models.longcat_client.client") as mock_client:
            mock_message = MagicMock()
            mock_message.content = '{"key": "value"}'
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = mock_message
            mock_client.chat.completions.create.return_value = mock_response
            result = chat_json(system="test", user="hi")
            assert result == {"key": "value"}

    def test_chat_json_with_markdown(self):
        from backend.models.longcat_client import chat_json
        with patch("backend.models.longcat_client.client") as mock_client:
            mock_message = MagicMock()
            mock_message.content = '```json\n{"key": "value"}\n```'
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = mock_message
            mock_client.chat.completions.create.return_value = mock_response
            result = chat_json(system="test", user="hi")
            assert result == {"key": "value"}

    def test_chat_json_invalid_raises(self):
        from backend.models.longcat_client import chat_json
        import json
        with patch("backend.models.longcat_client.client") as mock_client:
            mock_message = MagicMock()
            mock_message.content = "not json at all"
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = mock_message
            mock_client.chat.completions.create.return_value = mock_response
            with pytest.raises(json.JSONDecodeError):
                chat_json(system="test", user="hi")

    def test_chat_messages(self):
        from backend.models.longcat_client import chat_messages
        with patch("backend.models.longcat_client.client") as mock_client:
            mock_message = MagicMock()
            mock_message.content = "multi-message response"
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = mock_message
            mock_client.chat.completions.create.return_value = mock_response
            msgs = [{"role": "user", "content": "hello"}]
            result = chat_messages(msgs)
            assert result.content == "multi-message response"

    def test_chat_messages_with_tools(self):
        from backend.models.longcat_client import chat_messages
        with patch("backend.models.longcat_client.client") as mock_client:
            mock_message = MagicMock()
            mock_message.content = "resp"
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = mock_message
            mock_client.chat.completions.create.return_value = mock_response
            tools = [{"type": "function", "function": {"name": "t"}}]
            result = chat_messages([{"role": "user", "content": "hi"}], tools=tools)
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["tools"] == tools


class TestRetryLogic:
    def test_retry_on_rate_limit(self):
        """Test that _create_with_retry retries on RateLimitError"""
        from backend.models.longcat_client import _create_with_retry
        mock_response = MagicMock()
        with patch("backend.models.longcat_client.client") as mock_client, \
             patch("backend.models.longcat_client.LLM_RATE_LIMIT_RETRIES", 2), \
             patch("backend.models.longcat_client.LLM_RATE_LIMIT_WAIT", 0):
            # Fail twice then succeed
            rate_error = RateLimitError(
                message="rate limit",
                response=MagicMock(status_code=429),
                body=None
            )
            mock_client.chat.completions.create.side_effect = [
                rate_error, rate_error, mock_response
            ]
            result = _create_with_retry(model="test", messages=[])
            assert result == mock_response
            assert mock_client.chat.completions.create.call_count == 3

    def test_retry_exhausted_raises(self):
        """Test that _create_with_retry raises after all retries exhausted"""
        from backend.models.longcat_client import _create_with_retry
        with patch("backend.models.longcat_client.client") as mock_client, \
             patch("backend.models.longcat_client.LLM_RATE_LIMIT_RETRIES", 1), \
             patch("backend.models.longcat_client.LLM_RATE_LIMIT_WAIT", 0):
            rate_error = RateLimitError(
                message="rate limit",
                response=MagicMock(status_code=429),
                body=None
            )
            mock_client.chat.completions.create.side_effect = rate_error
            with pytest.raises(RateLimitError):
                _create_with_retry(model="test", messages=[])
            # Should have tried LLM_RATE_LIMIT_RETRIES + 1 = 2 times
            assert mock_client.chat.completions.create.call_count == 2

    def test_no_retry_on_success(self):
        """Test that _create_with_retry does not retry on success"""
        from backend.models.longcat_client import _create_with_retry
        mock_response = MagicMock()
        with patch("backend.models.longcat_client.client") as mock_client:
            mock_client.chat.completions.create.return_value = mock_response
            result = _create_with_retry(model="test", messages=[])
            assert result == mock_response
            assert mock_client.chat.completions.create.call_count == 1


class TestLoadPrompt:
    def test_load_prompt_existing_file(self, tmp_path):
        """Test load_prompt with an existing template file"""
        from backend.models.longcat_client import load_prompt
        # Create a temp prompt file
        agent_dir = tmp_path / "test_agent"
        agent_dir.mkdir()
        prompt_file = agent_dir / "greeting.txt"
        prompt_file.write_text("你好，我是{agent_name}", encoding="utf-8")
        with patch("backend.models.longcat_client._PROMPT_DIR", tmp_path):
            result = load_prompt("test_agent", "greeting")
            assert result == "你好，我是{agent_name}"

    def test_load_prompt_missing_file(self, tmp_path):
        """Test load_prompt returns empty string when file doesn't exist"""
        from backend.models.longcat_client import load_prompt
        with patch("backend.models.longcat_client._PROMPT_DIR", tmp_path):
            result = load_prompt("nonexistent", "missing")
            assert result == ""
