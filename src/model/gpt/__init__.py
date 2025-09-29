from .gpt import ask_chatgpt as _ask_chatgpt

# Auto-inject proxy from config unless caller explicitly sets it.
try:
    from src.utils.config import get_config  # available after config refactor
except Exception:
    get_config = None

def ask_chatgpt(api_key: str, model: str, user_message: str, prompt: str, proxy: str = "") -> tuple[bool, str]:
    if not proxy:
        try:
            if get_config:
                cfg = get_config()
                proxy = getattr(cfg.CHAT_GPT, "PROXY_FOR_CHAT_GPT", "") or ""
        except Exception:
            # fall back to empty proxy if config not ready
            proxy = proxy or ""
    return _ask_chatgpt(api_key, model, user_message, prompt, proxy=proxy)
