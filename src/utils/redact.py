
# src/utils/redact.py
# -*- coding: utf-8 -*-
import logging
import re

_RE_DISCORD_TOKEN = re.compile(r"[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}")
_RE_LONG_KEY = re.compile(r"\b[A-Za-z0-9]{32,}\b")
_RE_PROXY_AUTH = re.compile(r"(\w+://)([^:@\s]{2,}):([^@/\s]{2,})@")

def _redact_text(msg: str) -> str:
    if not msg:
        return msg
    s = str(msg)
    s = _RE_DISCORD_TOKEN.sub("***TOKEN***", s)
    s = _RE_LONG_KEY.sub(lambda m: f"{m.group(0)[:4]}…{m.group(0)[-4:]}", s)
    s = _RE_PROXY_AUTH.sub(lambda m: f"{m.group(1)}{m.group(2)[:2]}…:****@", s)
    return s

class SensitiveFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = _redact_text(record.msg)
            if record.args:
                new_args = []
                for a in record.args:
                    new_args.append(_redact_text(a) if isinstance(a, str) else a)
                record.args = tuple(new_args)
        except Exception:
            pass
        return True
