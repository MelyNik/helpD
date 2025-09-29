# apply_privacy_policy_patch.py
# Автоматически вносит правки приватности/политики в helpD:
# - .gitignore (секреты/логи/артефакты)
# - .env.example (ENV вместо секретов в файлах)
# - src/utils/redact.py (скрытие токенов/ключей в логах)
# - src/utils/config.py (оверлей из ENV)
# - main.py (Loguru -> redact patch)
# - policy engine: GM≤50% + mute 24h на 429/Verify
# - chatter: «слабые» реплаи (60%)
# - send*message*: mute на 429/Verify
import os, re, textwrap, pathlib, zipfile, shutil, sys

ROOT = pathlib.Path(__file__).parent.resolve()
REPORT = ROOT / "patch_report.txt"

def read(p): 
    try: return p.read_text(encoding="utf-8")
    except: return None

def write(p, s):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def append_unique(p, block):
    cur = read(p)
    if cur is None:
        write(p, block)
        return "created"
    if block.strip() in cur:
        return "kept"
    if not cur.endswith("\n"):
        cur += "\n"
    cur += block.rstrip()+"\n"
    write(p, cur); return "appended"

def backup_repo():
    dst = ROOT.with_name(ROOT.name+"_backup.zip")
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ROOT.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(ROOT))
    return dst

def apply_gitignore(report):
    gi = ROOT / ".gitignore"
    block = """\
### secrets & runtime artifacts
config.yaml
.env
.env.*
logs/
data/accounts.xlsx
data/state.json
*.log
*.sqlite
*.db
"""
    report.append(f".gitignore: {append_unique(gi, block)}")

def create_env_example(report):
    env = ROOT / ".env.example"
    content = textwrap.dedent(r"""
    # ====== Secrets & Paths ======
    # OpenAI / LLM keys (comma-separated if multiple)
    OPENAI_API_KEYS=
    
    # Proxy (optional). Format: http://user:pass@host:port
    HTTP_PROXY=
    HTTPS_PROXY=
    
    # Path to Discord accounts (keep OUTSIDE the repo)
    DISCORD_ACCOUNTS_PATH=D:\\secrets\\discord\\accounts.xlsx
    
    # Timezone for policy
    POLICY_TIMEZONE=Europe/Riga
    
    # ====== Feature toggles ======
    FEATURE_INVITER=0
    FEATURE_MESSAGE_DELETE=0
    FEATURE_VERIFY=0
    FEATURE_AI_CHATTER=1
    """).lstrip("\n")
    write(env, content); report.append(".env.example: created")

def create_redact_module(report):
    p = ROOT / "src" / "utils" / "redact.py"
    content = textwrap.dedent(r"""
    import re

    _DISCORD_TOKEN = re.compile(r"[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{6}\.[A-Za-z0-9_\-]{20,}")
    _LONG_SECRET   = re.compile(r"(?<![A-Za-z0-9])([A-Za-z0-9_\-]{32,})(?![A-Za-z0-9])")
    _KV_SECRET     = re.compile(r"(?i)\b(api[_-]?key|secret|token|pwd|password)\b\s*[:=]\s*(['\"]?)([^'\"\s]{8,})\2")
    _EMAIL         = re.compile(r"[A-Za-z0-9_.+\-]+@[A-Za-z0-9\-]+\.[A-Za-z0-9\-.]+")
    _BASIC_AUTH    = re.compile(r"(?i)(https?://)([^/:@\s]+):([^@/\s]+)@")

    def _mask_center(s: str) -> str:
        if len(s) <= 8:
            return "***"
        return f"{s[:4]}…{s[-4:]}"

    def redact_text(t: str) -> str:
        if not isinstance(t, str):
            return t
        t = _BASIC_AUTH.sub(r"\1***:***@", t)
        for rx in (_DISCORD_TOKEN, _LONG_SECRET):
            t = rx.sub(lambda m: _mask_center(m.group(0)), t)
        t = _KV_SECRET.sub(lambda m: f"{m.group(1)}={_mask_center(m.group(3))}", t)
        def _email(m):
            s = m.group(0); name, dom = s.split('@', 1)
            return (name[:1] + '…@' + dom)
        t = _EMAIL.sub(_email, t)
        return t
    """).lstrip("\n")
    write(p, content); report.append("src/utils/redact.py: created")

def patch_config_loader(report):
    p = ROOT / "src" / "utils" / "config.py"
    code = read(p)
    if code is None:
        content = textwrap.dedent(r"""
        import os, yaml
        from pathlib import Path

        def _get_env_list(name: str):
            v = os.getenv(name, "").strip()
            return [x.strip() for x in v.split(',') if x.strip()]

        def load_config(path: str = "config.yaml"):
            cfg = {}
            q = Path(path)
            if q.exists():
                with q.open('r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f) or {}
            ai = cfg.setdefault('CHAT_GPT', {})
            keys = _get_env_list('OPENAI_API_KEYS')
            if keys:
                ai['API_KEYS'] = keys
            proxy = os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY')
            if proxy:
                cfg.setdefault('NETWORK', {})['PROXY'] = proxy
            acc_path = os.getenv('DISCORD_ACCOUNTS_PATH')
            if acc_path:
                cfg.setdefault('DISCORD', {})['ACCOUNTS_PATH'] = acc_path
            tz = os.getenv('POLICY_TIMEZONE')
            if tz:
                cfg.setdefault('POLICY', {})['TIMEZONE'] = tz
            features = cfg.setdefault('FEATURES', {})
            for name in ('INVITER','MESSAGE_DELETE','VERIFY','AI_CHATTER'):
                v = os.getenv(f'FEATURE_{name}')
                if v is not None:
                    features[name] = v in ('1','true','True','yes','on')
            return cfg
        """).lstrip("\n")
        write(p, content); report.append("src/utils/config.py: created")
        return
    if "OPENAI_API_KEYS" in code and "_get_env_list" in code:
        report.append("src/utils/config.py: already has ENV overlay")
        return
    if "_get_env_list" not in code:
        code = "def _get_env_list(name: str):\n    import os\n    v = os.getenv(name, \"\").strip()\n    return [x.strip() for x in v.split(',') if x.strip()]\n\n" + code
    if "return cfg" in code:
        code = code.replace("return cfg", textwrap.dedent("""
            ai = cfg.setdefault('CHAT_GPT', {})
            keys = _get_env_list('OPENAI_API_KEYS')
            if keys:
                ai['API_KEYS'] = keys
            import os
            proxy = os.getenv('HTTPS_PROXY') or os.getenv('HTTP_PROXY')
            if proxy:
                cfg.setdefault('NETWORK', {})['PROXY'] = proxy
            acc_path = os.getenv('DISCORD_ACCOUNTS_PATH')
            if acc_path:
                cfg.setdefault('DISCORD', {})['ACCOUNTS_PATH'] = acc_path
            tz = os.getenv('POLICY_TIMEZONE')
            if tz:
                cfg.setdefault('POLICY', {})['TIMEZONE'] = tz
            features = cfg.setdefault('FEATURES', {})
            for name in ('INVITER','MESSAGE_DELETE','VERIFY','AI_CHATTER'):
                v = os.getenv(f'FEATURE_{name}')
                if v is not None:
                    features[name] = v in ('1','true','True','yes','on')
            return cfg
        """).strip("\n"))
    write(p, code); report.append("src/utils/config.py: patched with ENV overlay")

def patch_main(report):
    p = ROOT / "main.py"
    code = read(p)
    if code is None:
        report.append("main.py: NOT FOUND"); return
    if "redact_text" in code and "logger.patch" in code:
        report.append("main.py: already patched"); return
    if "from src.utils.redact import redact_text" not in code:
        code = "from src.utils.redact import redact_text\n" + code
    if "from loguru import logger" not in code:
        code = "from loguru import logger\n" + code
    if "_log_patch(" not in code:
        code = textwrap.dedent("""
        def _log_patch(record):
            try:
                msg = record.get("message")
                if isinstance(msg, str):
                    record["message"] = redact_text(msg)
                extra = record.get("extra", {})
                for k, v in list(extra.items()):
                    if isinstance(v, str):
                        extra[k] = redact_text(v)
                record["extra"] = extra
            except Exception:
                pass
            return record
        """) + "\n" + code
    if "logger.remove()" in code:
        code = code.replace("logger.remove()", "logger.remove()\nlogger = logger.patch(_log_patch)\nimport sys\nlogger.add(sys.stdout, format=\"<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}\")", 1)
    else:
        code = "logger.remove()\nlogger = logger.patch(_log_patch)\nimport sys\nlogger.add(sys.stdout, format=\"<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}\")\n" + code
    write(p, code); report.append("main.py: patched logger redact")

def patch_policy_engine(report):
    engines = list(ROOT.joinpath("src").rglob("engine.py"))
    if not engines:
        report.append("policy engine: NOT FOUND"); return
    p = engines[0]
    code = read(p) or ""
    changed = False
    if "gm_quota" not in code and "self.state = state_store.load()" in code:
        code = code.replace("self.state = state_store.load()", "self.state = state_store.load()\n        self.state.setdefault('gm_quota', {'day': None, 'gm': 0, 'total': 0})", 1)
        changed = True
    if "def on_red_flag" not in code:
        code += textwrap.dedent("""
        def on_red_flag(self, account_id: str, kind: str = "429"):
            m = self.state.setdefault('mute_until', {})
            from datetime import datetime, timedelta
            now = datetime.now(self._tz())
            until = now + timedelta(hours=24)
            m[str(account_id)] = int(until.timestamp())
            self._save_state()

        def is_muted(self, account_id: str) -> bool:
            ts = self.state.get('mute_until', {}).get(str(account_id))
            if not ts:
                return False
            import time
            return time.time() < ts

        def _today(self):
            from datetime import datetime
            return datetime.now(self._tz()).strftime('%Y-%m-%d')

        def _tz(self):
            import zoneinfo
            tz = self.cfg.get('POLICY', {}).get('TIMEZONE') or 'Europe/Riga'
            return zoneinfo.ZoneInfo(tz)

        def _is_gm(self, t: str) -> bool:
            head = t.lower().strip()
            return head.startswith(('gm', 'good morning', 'доброе утро'))

        def _replace_gm(self, t: str) -> str:
            repl = ['hey there','yo','hi','hello','привет','хэй']
            import random
            base = random.choice(repl)
            parts = t.split(maxsplit=1)
            return base if len(parts) < 2 else (base + ' ' + parts[1])
        """)
        changed = True
    if "def enforce_content" in code and "gm_quota" in code and "_is_gm" in code and "_replace_gm" in code and "_today" in code:
        if "gm' : 0" not in code and "gm_quota" in code and "replace_gm" in code:
            pass
    if "def enforce_content" in code and "gm_quota" not in code:
        code = re.sub(r"(def\s+enforce_content\s*\(self,\s*text:\s*str\)\s*->\s*str:\s*\n)",
                      r"""\1        t = text.strip()\n        today = self._today()\n        gmq = self.state['gm_quota']\n        if gmq['day'] != today:\n            gmq.update({'day': today, 'gm': 0, 'total': 0})\n        gmq['total'] += 1\n        if self._is_gm(t):\n            if gmq['gm'] * 2 >= gmq['total']:\n                t = self._replace_gm(t)\n            else:\n                gmq['gm'] += 1\n        # keep your other content rules here\n        return t\n""",
                      code, count=1)
        changed = True
    if changed:
        write(p, code); report.append(f"{p.relative_to(ROOT)}: patched (mute+gm quota)")
    else:
        report.append(f"{p.relative_to(ROOT)}: no changes applied")

def patch_chatter(report):
    files = list(ROOT.joinpath("src").rglob("chatter.py"))
    if not files:
        report.append("chatter.py: NOT FOUND"); return
    p = files[0]
    code = read(p) or ""
    if "is_weak" in code:
        report.append(f"{p.relative_to(ROOT)}: weak logic already present"); return
    if "policy.should_reply({" in code:
        code = code.replace("policy.should_reply({",
                            "strong = bool(getattr(msg, 'is_reply_to_us', False) or getattr(msg, 'mentions_us', False))\n"
                            "weak_trigger = (not strong) and (('?' in getattr(msg, 'text', '')) or (\n"
                            "    hasattr(msg, 'matches_topics') and msg.matches_topics(self.cfg.get('TOPICS', []))\n"
                            "))\n"
                            "policy.should_reply({")
        code = code.replace("})", ", 'is_weak': weak_trigger })", 1)
        write(p, code); report.append(f"{p.relative_to(ROOT)}: injected weak-trigger param")
    else:
        report.append(f"{p.relative_to(ROOT)}: could not locate should_reply call")

def patch_sender(report):
    cands = list(ROOT.joinpath("src").rglob("*send*message*.py"))
    if not cands:
        report.append("send message module: NOT FOUND"); return
    p = cands[0]
    code = read(p) or ""
    if "mute 24h" in code or "on_red_flag" in code:
        report.append(f"{p.relative_to(ROOT)}: mute logic seems present"); return
    if "async def" in code and "send" in code:
        code = re.sub(r"(async\s+def\s+\w+\(.*?\):\s*\n)",
                      r"""\1    # Mute check before sending\n    try:\n        if hasattr(self, 'policy') and hasattr(self, 'account') and self.policy.is_muted(getattr(self.account, 'index', '0')):\n            from loguru import logger as _lg\n            _lg.info(f"[mute] skip send: account={getattr(self.account, 'index', '0')}")\n            return None\n    except Exception:\n        pass\n""",
                      code, count=1, flags=re.DOTALL)
        if "429" not in code or "verify" not in code.lower():
            code += textwrap.dedent("""

            async def _handle_send_exception(e):
                status = getattr(e, 'status', None)
                body = getattr(e, 'text', '') or str(e)
                if status == 429 or ('captcha' in body.lower()) or ('verify' in body.lower()):
                    try:
                        self.policy.on_red_flag(account_id=str(getattr(self.account, 'index', '0')), kind=str(status or 'verify'))
                        from loguru import logger as _lg
                        _lg.warning(f"[red-flag] {status} -> mute 24h")
                        return None
                    except Exception:
                        pass
                raise e
            """)
        write(p, code); report.append(f"{p.relative_to(ROOT)}: added mute-on-429/verify hooks")
    else:
        report.append(f"{p.relative_to(ROOT)}: could not identify async send function")

def main():
    report = []
    backup = backup_repo()
    report.append(f"Backup created: {backup.name}")
    apply_gitignore(report)
    create_env_example(report)
    create_redact_module(report)
    patch_config_loader(report)
    patch_main(report)
    patch_policy_engine(report)
    patch_chatter(report)
    patch_sender(report)
    (ROOT / "patch_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("Done. See patch_report.txt")

if __name__ == "__main__":
    main()
