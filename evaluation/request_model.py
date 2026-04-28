import os
from functools import lru_cache


RETRY_TRIES = 3
RETRY_DELAY = 1
RETRY_BACKOFF = 2
RETRY_JITTER = (0, 0.3)


def _get_openai_module():
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "The `openai` package is required for `--backend api`. "
            "Install dependencies from `requirements.txt`."
        ) from exc
    return openai


def _get_retry_decorator():
    try:
        from retry import retry
    except ImportError as exc:
        raise ImportError(
            "The `retry` package is required for `--backend api`. "
            "Install dependencies from `requirements.txt`."
        ) from exc
    return retry


def _get_retry_exceptions():
    openai = _get_openai_module()

    exception_names = [
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    ]
    exceptions = tuple(
        getattr(openai, name)
        for name in exception_names
        if hasattr(openai, name)
    )
    return exceptions or (Exception,)


def _normalize_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return str(content)


@lru_cache(maxsize=1)
def _get_client():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for `--backend api`.")

    openai = _get_openai_module()
    client_kwargs = {}

    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url

    return openai.OpenAI(**client_kwargs)


def _build_retrying_request():
    retry = _get_retry_decorator()
    retry_exceptions = _get_retry_exceptions()

    @retry(
        exceptions=retry_exceptions,
        tries=RETRY_TRIES,
        delay=RETRY_DELAY,
        backoff=RETRY_BACKOFF,
        jitter=RETRY_JITTER,
    )
    def _request(model_key, messages, top_p, temperature, timeout):
        response = _get_client().chat.completions.create(
            model=model_key,
            messages=messages,
            top_p=top_p,
            temperature=temperature,
            timeout=timeout,
        )
        return response

    return _request


def request_model_by_model_key(
    model_key,
    messages,
    top_p,
    temperature,
    top_k=None,
    timeout=300,
):
    response = _build_retrying_request()(
        model_key=model_key,
        messages=messages,
        top_p=top_p,
        temperature=temperature,
        timeout=timeout,
    )

    message = response.choices[0].message
    content = _normalize_content(getattr(message, "content", ""))

    return {
        "output": {
            "choices": [
                {
                    "message": {
                        "content": content,
                    }
                }
            ]
        }
    }
