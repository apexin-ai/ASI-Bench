"""
测试三个模型的 API 可用性：Claude Opus 4.6、GPT-5.4、Gemini 3.1 Pro Preview

- GPT-5.4: 通过 OpenAI API 直接调用
- Claude Opus 4.6: 通过 OpenRouter 调用
- Gemini 3.1 Pro Preview: 通过 OpenRouter 调用
"""

import os
import pytest
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI

load_dotenv()

pytestmark = pytest.mark.integration

TEST_PROMPT = "Say 'hello' in one word."


@pytest.fixture(scope="module")
def openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


@pytest.fixture(scope="module")
def openrouter_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def _call_model(client: OpenAI, model: str, use_max_completion_tokens: bool = False) -> str:
    """调用模型并返回响应文本"""
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": TEST_PROMPT}],
    )
    # thinking 模型（如 Gemini 3.1 Pro）的 reasoning tokens 也计入 max_completion_tokens，
    # 所以需要设置更大的值以确保有足够空间输出内容
    token_limit = 256 if use_max_completion_tokens else 32
    if use_max_completion_tokens:
        kwargs["max_completion_tokens"] = token_limit
    else:
        kwargs["max_tokens"] = token_limit
    try:
        response = client.chat.completions.create(**kwargs)
    except (APIConnectionError, APITimeoutError) as exc:
        pytest.skip(f"Model API endpoint unavailable from this environment: {exc}")
    content = response.choices[0].message.content
    assert content is not None and len(content.strip()) > 0, "模型返回了空响应"
    print(f"\n[{model}] 响应: {content.strip()}")
    return content


class TestGPT54:
    """测试 GPT-5.4 (OpenAI API)"""

    def test_gpt54_responds(self, openai_client):
        content = _call_model(openai_client, "gpt-5.4", use_max_completion_tokens=True)
        assert "hello" in content.lower()


class TestClaudeOpus46:
    """测试 Claude Opus 4.6 (OpenRouter)"""

    def test_claude_opus_responds(self, openrouter_client):
        content = _call_model(openrouter_client, "anthropic/claude-opus-4-6")
        assert "hello" in content.lower()


class TestGemini31ProPreview:
    """测试 Gemini 3.1 Pro Preview (OpenRouter)"""

    def test_gemini_responds(self, openrouter_client):
        content = _call_model(openrouter_client, "google/gemini-3.1-pro-preview", use_max_completion_tokens=True)
        assert "hello" in content.lower()
