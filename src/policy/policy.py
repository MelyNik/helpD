
# src/policy/policy.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import random
import re
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

_WORD_RE = re.compile(r"[^\s]+", re.UNICODE)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_CHANNEL_MENTION_RE = re.compile(r"<#\d+>")
_EMOJI_RE = re.compile(r"([\U0001F1E6-\U0001F1FF]|[\U0001F300-\U0001FAFF]|[\U00002700-\U000027BF])")

REACTION_PACK = ("thumbsup", "eyes", "white_check_mark")


@dataclass
class PolicyClock:
    tz: str
    active_start: int
    active_end: int

    def now(self) -> datetime:
        if ZoneInfo:
            return datetime.now(ZoneInfo(self.tz))
        return datetime.utcnow()

    def is_active_hours(self, dt: Optional[datetime] = None) -> bool:
        dt = dt or self.now()
        h = dt.hour
        start, end = self.active_start, self.active_end
        if start <= end:
            return start <= h < end
        return h >= start or h < end


class PolicyEngine:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("POLICY", {})
        st = config.get("STATE", {})
        self.state_path = st.get("PATH", "data/state.json")

        tz = self.cfg.get("TIMEZONE", "Asia/Kolkata")
        active_hours = self.cfg.get("ACTIVE_HOURS", [9, 23])
        self.clock = PolicyClock(tz=tz, active_start=int(active_hours[0]), active_end=int(active_hours[1]))

        slots = self.cfg.get("INITIATIVE_SLOTS", {})
        self.slot_start_hour = int(slots.get("START_HOUR", 9))
        self.slot_end_hour = int(slots.get("END_HOUR", 23))
        self.slot_count = int(slots.get("COUNT", 6))
        self.slot_start_prob = float(slots.get("START_PROB", 0.60))
        self.slot_min_prob = float(slots.get("MIN_PROB", 0.20))
        self.slot_max_prob = float(slots.get("MAX_PROB", 0.80))
        self.slot_step = float(slots.get("STEP", 0.10))
        self.daily_target_min = int(slots.get("DAILY_TARGET_MIN", 4))
        self.daily_target_max = int(slots.get("DAILY_TARGET_MAX", 6))
        self.low_replies_threshold = int(slots.get("LOW_REPLIES_THRESHOLD_PER_DAY", 2))
        self.low_replies_days = int(slots.get("LOW_REPLIES_CONSECUTIVE_DAYS", 2))
        self.temp_target_on_low = int(slots.get("TEMP_TARGET_ON_LOW_REPLIES", 6))

        rp = self.cfg.get("REPLY_POLICY", {})
        self.strong_reply_prob = float(rp.get("STRONG_REPLY_PROB", 0.90))
        self.weak_reply_prob = float(rp.get("WEAK_REPLY_PROB", 0.60))
        self.reply_within_minutes = int(rp.get("REPLY_WITHIN_MINUTES", 3))
        self.thread_max_actions = int(rp.get("THREAD_MAX_ACTIONS", 2))

        pauses = self.cfg.get("PAUSES", {})
        self.between_init_min = int(pauses.get("BETWEEN_INIT_MIN_MINUTES", 45))
        self.between_init_max = int(pauses.get("BETWEEN_INIT_MAX_MIN_MINUTES", self.between_init_min + 60)) if False else int(pauses.get("BETWEEN_INIT_MAX_MINUTES", 120))
        self.min_gap_after_any = int(pauses.get("MIN_GAP_AFTER_ANY_MESSAGE_MINUTES", 2))

        cnt = self.cfg.get("CONTENT", {})
        self.min_words = int(cnt.get("MIN_WORDS", 3))
        self.max_words = int(cnt.get("MAX_WORDS", 11))
        self.no_links = bool(cnt.get("NO_LINKS", True))
        self.no_channel_mentions = bool(cnt.get("NO_CHANNEL_MENTIONS", True))
        self.no_emoji_in_text = bool(cnt.get("NO_EMOJI_IN_TEXT", True))
        self.greetings_per_lang = int(cnt.get("GREETINGS_PER_LANG", 7))
        self.gm_ratio_max = float(cnt.get("GM_RATIO_MAX", 0.50))

        dd = self.cfg.get("DEDUP", {})
        self.dedup_window_hours = int(dd.get("WINDOW_HOURS", 24))

        rf = self.cfg.get("RED_FLAGS", {})
        self.silence_hours = int(rf.get("SILENCE_HOURS", 24))

        self.per_channel_limits = {str(k): v for k, v in self.cfg.get("PER_CHANNEL_LIMITS", {}).items()}

        self._lock = threading.Lock()
        self.state: Dict[str, Any] = self._load_state()

    # Public API
    def should_initiate(self, account_id: str, channel_id: str, now: Optional[datetime] = None) -> bool:
        now = now or self.clock.now()
        with self._lock:
            if not self._is_account_active(account_id, now):
                return False
            if not self.clock.is_active_hours(now):
                return False
            self._tick_day_reset_if_needed(account_id, now)
            target_max = self._compute_daily_target_max(account_id)
            if self._get_daily(account_id, "initiatives") >= target_max:
                return False
            if not self._under_channel_cap(account_id, channel_id):
                return False
            if not self._gap_ok(account_id, now, gap_minutes=self.between_init_min):
                return False
            slot_idx = self._slot_index(now)
            self._on_slot_transition(account_id, now, slot_idx)
            key = self._acc_key(account_id)
            checked_slots = self.state[key]["slots_checked"]
            if self._slot_key(now, slot_idx) in checked_slots:
                return False
            checked_slots.add(self._slot_key(now, slot_idx))
            p = self.state[key]["slot_probs"][slot_idx]
            return bool(random.random() < p)

    def should_reply(self, ctx: Dict[str, Any]) -> Tuple[Optional[str], Any]:
        account_id = str(ctx.get("account"))
        channel_id = str(ctx.get("channel"))
        thread_id = str(ctx.get("thread_id")) if ctx.get("thread_id") else None
        is_strong = bool(ctx.get("is_strong", False))
        now = ctx.get("now") or self.clock.now()

        with self._lock:
            if not self._is_account_active(account_id, now):
                return (None, None)
            if not self.clock.is_active_hours(now):
                return (None, None)
            if not self._gap_ok(account_id, now, gap_minutes=self.min_gap_after_any):
                return (None, None)
            if thread_id and self._thread_actions(account_id, thread_id) >= self.thread_max_actions:
                return (None, None)

            p = self.strong_reply_prob if is_strong else self.weak_reply_prob
            if random.random() >= p:
                return (None, None)

            reaction_ratio = 0.10 if is_strong else 0.20
            if random.random() < reaction_ratio:
                return ("reaction", random.choice(REACTION_PACK))
            return ("text", None)

    def enforce_content(self, text: str, lang_hint: Optional[str] = None) -> str:
        t = (text or "").strip()
        if self.no_links:
            t = _URL_RE.sub("", t)
        if self.no_channel_mentions:
            t = _CHANNEL_MENTION_RE.sub("", t)
        if self.no_emoji_in_text:
            t = _EMOJI_RE.sub("", t)
        t = re.sub(r"\s+", " ", t).strip()
        words = t.split()
        if not words:
            return "ok"
        if len(words) > self.max_words:
            t = " ".join(words[: self.max_words])
        return t

    def is_allowed_text(self, account_id: str, channel_id: str, thread_id: Optional[str], text: str) -> bool:
        place = self._place_key(channel_id, thread_id)
        norm = self._norm_text(text)
        with self._lock:
            self._gc_dedup(now=self.clock.now())
            for old_norm, _ts in self.state["dedup"][place]:
                if self._similar(norm, old_norm):
                    return False
        return True

    def record_event(self, event: Dict[str, Any]) -> None:
        etype = event.get("type")
        account_id = str(event.get("account"))
        channel_id = str(event.get("channel"))
        thread_id = str(event.get("thread")) if event.get("thread") else None
        text = event.get("text", "")
        now = self.clock.now()

        with self._lock:
            self._tick_day_reset_if_needed(account_id, now)
            key = self._acc_key(account_id)
            self.state[key]["last_message_ts"] = now.timestamp()

            if etype == "initiative":
                self._incr_daily(account_id, "initiatives", 1)
                self._incr_channel_daily(account_id, channel_id, 1)
                slot_idx = self._slot_index(now)
                self.state[key]["slots_fired"].add(self._slot_key(now, slot_idx))
                self.state[key]["last_slot_idx"] = slot_idx
                self.state[key]["last_slot_date"] = now.date().isoformat()
            elif etype == "reply_text":
                self._incr_daily(account_id, "replies_text", 1)
                if thread_id:
                    self._incr_thread_actions(account_id, thread_id)
            elif etype == "reply_reaction":
                self._incr_daily(account_id, "replies_reaction", 1)
                if thread_id:
                    self._incr_thread_actions(account_id, thread_id)

            if text:
                place = self._place_key(channel_id, thread_id)
                self.state["dedup"][place].append((self._norm_text(text), now.timestamp()))
                if len(self.state["dedup"][place]) > 200:
                    self.state["dedup"][place].popleft()

            self._save_state()

    def on_red_flag(self, account_id: str, kind: str) -> None:
        now = self.clock.now()
        until = now + timedelta(hours=self.silence_hours)
        with self._lock:
            self.state[self._acc_key(account_id)]["silence_until_ts"] = until.timestamp()
            self._save_state()

    # Internal helpers
    def _load_state(self) -> Dict[str, Any]:
        dirn = os.path.dirname(self.state_path) or "."
        os.makedirs(dirn, exist_ok=True)
        if os.path.isfile(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in list(data.get("dedup", {}).items()):
                        data["dedup"][k] = deque(v, maxlen=400)
                    return data
            except Exception:
                pass
        return {"accounts": {}, "dedup": defaultdict(lambda: deque(maxlen=400))}

    def _save_state(self) -> None:
        tmp = {
            "accounts": self.state.get("accounts", {}),
            "dedup": {k: list(v) for k, v in self.state.get("dedup", {}).items()},
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(tmp, f, ensure_ascii=False)

    def _acc_key(self, account_id: str) -> str:
        accs = self.state["accounts"]
        if account_id not in accs:
            accs[account_id] = self._new_account_state()
        return account_id

    def _new_account_state(self) -> Dict[str, Any]:
        probs = [self.slot_start_prob for _ in range(self.slot_count)]
        return {
            "last_message_ts": 0.0,
            "silence_until_ts": 0.0,
            "day": "",
            "daily": {"initiatives": 0, "replies_text": 0, "replies_reaction": 0},
            "per_channel_daily": {},
            "slot_probs": probs,
            "slots_checked": set(),
            "slots_fired": set(),
            "last_slot_idx": None,
            "last_slot_date": "",
            "low_reply_history": deque(maxlen=7),
            "threads": {},
        }

    def _tick_day_reset_if_needed(self, account_id: str, now: datetime) -> None:
        key = self._acc_key(account_id)
        acc = self.state["accounts"][key]
        today = now.date().isoformat()
        if acc["day"] != today:
            if acc["day"]:
                replies_yest = acc["daily"]["replies_text"] + acc["daily"]["replies_reaction"]
                acc["low_reply_history"].append(int(replies_yest))
            acc["day"] = today
            acc["daily"] = {"initiatives": 0, "replies_text": 0, "replies_reaction": 0}
            acc["per_channel_daily"] = {}
            acc["slots_checked"] = set()
            acc["slots_fired"] = set()
            acc["slot_probs"] = [self.slot_start_prob for _ in range(self.slot_count)]
            acc["last_slot_idx"] = None
            acc["last_slot_date"] = today

    def _compute_daily_target_max(self, account_id: str) -> int:
        key = self._acc_key(account_id)
        acc = self.state["accounts"][key]
        hist = list(acc["low_reply_history"])
        if len(hist) >= self.low_replies_days and all(x < self.low_replies_threshold for x in hist[-self.low_replies_days :]):
            return max(self.daily_target_min, self.temp_target_on_low)
        return self.daily_target_max

    def _under_channel_cap(self, account_id: str, channel_id: str) -> bool:
        cap_cfg = self.per_channel_limits.get(str(channel_id))
        if not cap_cfg:
            return True
        daily_max = int(cap_cfg.get("daily_initiatives_max", 999999))
        done = self._get_channel_daily(account_id, channel_id)
        return done < daily_max

    def _gap_ok(self, account_id: str, now: datetime, gap_minutes: int) -> bool:
        key = self._acc_key(account_id)
        last_ts = float(self.state["accounts"][key]["last_message_ts"] or 0.0)
        if last_ts <= 0:
            return True
        return (now - datetime.fromtimestamp(last_ts, tz=now.tzinfo)) >= timedelta(minutes=gap_minutes)

    def _slot_index(self, now: datetime) -> int:
        span = (self.slot_end_hour - self.slot_start_hour) % 24
        if span == 0:
            span = 24
        slot_len = span / float(self.slot_count)
        h = now.hour + now.minute / 60.0
        rel = (h - self.slot_start_hour) % 24
        idx = int(rel // slot_len)
        if idx < 0:
            idx = 0
        if idx >= self.slot_count:
            idx = self.slot_count - 1
        return idx

    def _slot_key(self, now: datetime, idx: int) -> str:
        return f"{now.date().isoformat()}#{idx}"

    def _on_slot_transition(self, account_id: str, now: datetime, current_idx: int) -> None:
        key = self._acc_key(account_id)
        acc = self.state["accounts"][key]
        last_idx = acc["last_slot_idx"]
        last_date = acc["last_slot_date"]
        today = now.date().isoformat()
        if last_idx is None:
            acc["last_slot_idx"] = current_idx
            acc["last_slot_date"] = today
            return
        if last_date != today:
            acc["last_slot_idx"] = current_idx
            acc["last_slot_date"] = today
            return
        if last_idx == current_idx:
            return

        fired_prev = (f"{today}#{last_idx}" in acc["slots_fired"])
        cur_p = acc["slot_probs"][current_idx]
        if fired_prev:
            cur_p = max(self.slot_min_prob, cur_p - self.slot_step)
        else:
            cur_p = min(self.slot_max_prob, cur_p + self.slot_step)
        acc["slot_probs"][current_idx] = cur_p
        acc["last_slot_idx"] = current_idx
        acc["last_slot_date"] = today

    def _thread_actions(self, account_id: str, thread_id: str) -> int:
        key = self._acc_key(account_id)
        return int(self.state["accounts"][key]["threads"].get(thread_id, 0))

    def _incr_thread_actions(self, account_id: str, thread_id: str) -> None:
        key = self._acc_key(account_id)
        th = self.state["accounts"][key]["threads"]
        th[thread_id] = int(th.get(thread_id, 0)) + 1

    def _incr_daily(self, account_id: str, field: str, n: int) -> None:
        key = self._acc_key(account_id)
        self.state["accounts"][key]["daily"][field] += int(n)

    def _get_daily(self, account_id: str, field: str) -> int:
        key = self._acc_key(account_id)
        return int(self.state["accounts"][key]["daily"].get(field, 0))

    def _incr_channel_daily(self, account_id: str, channel_id: str, n: int) -> None:
        key = self._acc_key(account_id)
        m = self.state["accounts"][key]["per_channel_daily"]
        m[str(channel_id)] = int(m.get(str(channel_id), 0)) + int(n)

    def _get_channel_daily(self, account_id: str, channel_id: str) -> int:
        key = self._acc_key(account_id)
        m = self.state["accounts"][key]["per_channel_daily"]
        return int(m.get(str(channel_id), 0))

    def _is_account_active(self, account_id: str, now: datetime) -> bool:
        key = self._acc_key(account_id)
        until = float(self.state["accounts"][key]["silence_until_ts"] or 0.0)
        return now.timestamp() >= until

    def _gc_dedup(self, now: Optional[datetime] = None) -> None:
        now = now or self.clock.now()
        cutoff = now - timedelta(hours=self.dedup_window_hours)
        for place, q in list(self.state["dedup"].items()):
            while q and datetime.fromtimestamp(q[0][1], tz=now.tzinfo) < cutoff:
                q.popleft()

    def _place_key(self, channel_id: str, thread_id: Optional[str]) -> str:
        return f"{channel_id}#{thread_id or '-'}"

    def _norm_text(self, t: str) -> str:
        t = t.lower().strip()
        t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
        t = re.sub(r"\s+", " ", t)
        return t

    def _similar(self, a: str, b: str) -> bool:
        if a == b:
            return True
        if len(a) >= 6 and (a in b or b in a):
            return True
        return False
