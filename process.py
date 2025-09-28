import asyncio
import random

from loguru import logger

from src.model import prepare_data
import src.utils
from src.utils.output import show_dev_info, show_logo, show_menu
from src.utils.reader import read_xlsx_accounts
from src.utils.constants import ACCOUNTS_FILE, Account
import src.model
from src.utils.check_github_version import check_version




async def start():
    async def launch_wrapper(account):
        async with semaphore:
            await account_flow(account, config)

    show_logo()
    show_dev_info()

    print("")

    config = src.utils.get_config()

    # === Policy & log-safety init (added) ===
    from src.policy.policy import PolicyEngine
    from src.utils.redact import SensitiveFilter
    import logging
    logging.getLogger().addFilter(SensitiveFilter())
    # Build a plain dict for PolicyEngine (supports dict-like config or object with __dict__)
    try:
        cfg_dict = config.__dict__ if hasattr(config, "__dict__") else dict(config)
    except Exception:
        cfg_dict = {}
    # Merge in POLICY defaults if absent
    cfg_dict.setdefault("POLICY", {
        "TIMEZONE": "Asia/Kolkata",
        "ACTIVE_HOURS": [9, 23],
        "INITIATIVE_SLOTS": {
            "START_HOUR": 9, "END_HOUR": 23, "COUNT": 6,
            "START_PROB": 0.60, "MIN_PROB": 0.20, "MAX_PROB": 0.80, "STEP": 0.10,
            "DAILY_TARGET_MIN": 4, "DAILY_TARGET_MAX": 6,
            "LOW_REPLIES_THRESHOLD_PER_DAY": 2, "LOW_REPLIES_CONSECUTIVE_DAYS": 2,
            "TEMP_TARGET_ON_LOW_REPLIES": 6
        },
        "REPLY_POLICY": {
            "STRONG_REPLY_PROB": 0.90, "WEAK_REPLY_PROB": 0.60,
            "REPLY_WITHIN_MINUTES": 3, "THREAD_MAX_ACTIONS": 2
        },
        "PAUSES": {
            "BETWEEN_INIT_MIN_MINUTES": 45, "BETWEEN_INIT_MAX_MINUTES": 120,
            "MIN_GAP_AFTER_ANY_MESSAGE_MINUTES": 2
        },
        "CONTENT": {
            "MIN_WORDS": 3, "MAX_WORDS": 11,
            "NO_LINKS": True, "NO_CHANNEL_MENTIONS": True, "NO_EMOJI_IN_TEXT": True,
            "GREETINGS_PER_LANG": 7, "GM_RATIO_MAX": 0.50
        },
        "DEDUP": {"WINDOW_HOURS": 24},
        "RED_FLAGS": {"SILENCE_HOURS": 24},
        "PER_CHANNEL_LIMITS": {}
    })
    cfg_dict.setdefault("STATE", {"PATH": "data/state.json"})
    policy = PolicyEngine(cfg_dict)
    # Expose policy to config so deeper modules can use it
    try:
        config.POLICY_ENGINE = policy
    except Exception:
        pass
    # === end policy init ===

    task = show_menu(src.utils.constants.MAIN_MENU_OPTIONS)
    if task == "Exit":
        return

    config.DATA_FOR_TASKS = await prepare_data(config, task)

    config.TASK = task

    # Читаем аккаунты из XLSX
    all_accounts = read_xlsx_accounts(ACCOUNTS_FILE)

    # Определяем диапазон аккаунтов
    start_index = config.SETTINGS.ACCOUNTS_RANGE[0]
    end_index = config.SETTINGS.ACCOUNTS_RANGE[1]

    # Если оба 0, проверяем EXACT_ACCOUNTS_TO_USE
    if start_index == 0 and end_index == 0:
        if config.SETTINGS.EXACT_ACCOUNTS_TO_USE:
            # Фильтруем аккаунты по конкретным номерам
            accounts_to_process = [
                acc
                for acc in all_accounts
                if acc.index in config.SETTINGS.EXACT_ACCOUNTS_TO_USE
            ]
            logger.info(
                f"Using specific accounts: {config.SETTINGS.EXACT_ACCOUNTS_TO_USE}"
            )
        else:
            # Если список пустой, берем все аккаунты
            accounts_to_process = all_accounts
    else:
        # Фильтруем аккаунты по диапазону
        accounts_to_process = [
            acc for acc in all_accounts if start_index <= acc.index <= end_index
        ]

    if not accounts_to_process:
        logger.error("No accounts found in specified range")
        return

    # Проверяем наличие прокси
    if not any(account.proxy for account in accounts_to_process):
        logger.error("No proxies found in accounts data")
        return

    threads = config.SETTINGS.THREADS

    # Создаем список аккаунтов и перемешиваем его
    if config.SETTINGS.SHUFFLE_ACCOUNTS:
        shuffled_accounts = list(accounts_to_process)
        random.shuffle(shuffled_accounts)
    else:
        shuffled_accounts = accounts_to_process

    # Создаем строку с порядком аккаунтов
    account_order = " ".join(str(acc.index) for acc in shuffled_accounts)
    logger.info(
        f"Starting with accounts {min(acc.index for acc in accounts_to_process)} "
        f"to {max(acc.index for acc in accounts_to_process)}..."
    )
    logger.info(f"Accounts order: {account_order}")

    semaphore = asyncio.Semaphore(value=threads)
    tasks = []

    # Создаем задачи для каждого аккаунта
    for account in shuffled_accounts:
        tasks.append(asyncio.create_task(launch_wrapper(account)))

    await asyncio.gather(*tasks)


async def account_flow(account: Account, config: src.utils.config.Config):
    try:
        pause = random.randint(
            config.SETTINGS.RANDOM_INITIALIZATION_PAUSE[0],
            config.SETTINGS.RANDOM_INITIALIZATION_PAUSE[1],
        )
        logger.info(f"[{account.index}] Sleeping for {pause} seconds before start...")
        await asyncio.sleep(pause)

        instance = src.model.Start(account, config)

        await wrapper(instance.initialize, config)

        await wrapper(instance.flow, config)

        pause = random.randint(
            config.SETTINGS.RANDOM_PAUSE_BETWEEN_ACCOUNTS[0],
            config.SETTINGS.RANDOM_PAUSE_BETWEEN_ACCOUNTS[1],
        )
        logger.info(f"Sleeping for {pause} seconds before next account...")
        await asyncio.sleep(pause)

    except Exception as err:
        logger.error(f"{account.index} | Account flow failed: {err}")


async def wrapper(function, config: src.utils.config.Config, *args, **kwargs):
    attempts = config.SETTINGS.ATTEMPTS
    for attempt in range(attempts):
        result = await function(*args, **kwargs)
        if isinstance(result, tuple) and result and isinstance(result[0], bool):
            if result[0]:
                return result
        elif isinstance(result, bool):
            if result:
                return True

        if attempt < attempts - 1:  # Don't sleep after the last attempt
            pause = random.randint(
                config.SETTINGS.PAUSE_BETWEEN_ATTEMPTS[0],
                config.SETTINGS.PAUSE_BETWEEN_ATTEMPTS[1],
            )
            logger.info(
                f"Sleeping for {pause} seconds before next attempt {attempt+1}/{config.SETTINGS.ATTEMPTS}..."
            )
            await asyncio.sleep(pause)

    return result


def task_exists_in_config(task_name: str, tasks_list: list) -> bool:
    """Рекурсивно проверяет наличие задачи в списке задач, включая вложенные списки"""
    for task in tasks_list:
        if isinstance(task, list):
            if task_exists_in_config(task_name, task):
                return True
        elif task == task_name:
            return True
    return False
