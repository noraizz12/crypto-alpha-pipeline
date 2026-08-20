import asyncio
import functools
import hashlib
import os
import sys
import random
import time
import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import psutil
import opsgenie_sdk
import requests
from .aws import load_aws_secrets
from .directory import LOG_DIR
from .util import log_and_raise

CRITICAL = "critical"
HIGH = "high"
MEDIUM = "medium"
LOW = "low"

# Load OpsGenie API key - gracefully handle missing credentials
OPS_GENIE_API_KEY = os.environ.get('OPSGENIE_API_KEY', '')
if not OPS_GENIE_API_KEY:
    try:
        secrets = load_aws_secrets(statarb_secretid='statarb/ops_genie_api_key')
        if secrets:
            OPS_GENIE_API_KEY = secrets.get('ops_genie_api_key', '')
    except Exception as e:
        logger.warning(f"OpsGenie API key not available: {e}. Alerting disabled.")

PRIORITY_TO_OPSGENIE_PRI = {CRITICAL: "P1", HIGH: "P2", MEDIUM: "P3", LOW: "P4"}

SECONDS_IN_MINUTE = 60
MIN_WAIT_RAISE_ALERT = 1 * SECONDS_IN_MINUTE

DEFAULT_NUM_RETRIES = 3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AlertAction(str, Enum):
    BOOTSTRAP = "bootstrap"
    RAISE = "raise"
    CLEAR = "clear"
    UPDATE_DESCRIPTION = "update_description"
    UPDATE_PRIORITY = "update_priority"
    ADD_NOTE = "add_note"


def alert_key_hash(key: str) -> str:
    return hashlib.md5(str(key).encode()).hexdigest()


def is_opsgenie_alerting_disabled() -> Optional[str]:
    return os.environ.get("DISABLE_ALERTS")


class OpsGenie:
    _executor = None
    instance = None
    _loop = None

    @classmethod
    def singleton(cls):
        if not OpsGenie.instance:
            OpsGenie.instance = cls(OPS_GENIE_API_KEY)
        return OpsGenie.instance

    @classmethod
    def get_executor(cls):
        if cls._executor is None:
            cls._executor = ThreadPoolExecutor(5)
        return cls._executor

    @classmethod
    def set_loop(cls, loop):
        cls._loop = loop

    @classmethod
    def get_loop(cls):
        if cls._loop is None:
            cls._loop = asyncio.get_event_loop()
        return cls._loop

    def __init__(self, opsgenie_api_key: str):
        self.active_alert_keys = set()
        self.active_alert_key_to_update_tstamp = {}
        self.alert_key_to_last_priority = {}
        self.alert_key_to_last_raise_tstamp = {}
        self.alert_key_to_last_clear_tstamp = {}
        self.alert_key_to_last_close_req = {}
        self.executor = self.get_executor()
        self.conf = opsgenie_sdk.configuration.Configuration()
        self.conf.api_key["Authorization"] = opsgenie_api_key
        self.api_client = opsgenie_sdk.api_client.ApiClient(configuration=self.conf, pool_threads=5)
        self.alert_api = opsgenie_sdk.AlertApi(api_client=self.api_client)

        self.bootstrapped = False
        self.alerts_filename = LOG_DIR + "/active_alerts.json"
        self.tasks_currently_running = 0
        self._bootstrap_lock = asyncio.Lock()
        self.details = {
            "Location": "statarb",
            "Process": f"{' '.join(psutil.Process(os.getpid()).cmdline())}",
            "StatArb": True,
        }
        self.details_str = "\n".join(f"{k}: {v}" for k, v in self.details.items())
        self.disabled_alert_keys = set()
        self.disable_forever_bootstrap = True
        self._bootstrap_task = self.get_loop().create_task(self.bootstrap_from_local())
        logger.info("Init opsgenie object")

    async def execute_call_with_retry(self, api_call, action: AlertAction, key: Optional[str] = None):
        """
        Execute ops genie api request in thread executor
        """
        self.tasks_currently_running += 1
        for attempt_count in range(DEFAULT_NUM_RETRIES):
            if attempt_count > 0:
                await asyncio.sleep(2**attempt_count)
            elif action == AlertAction.RAISE:
                # Default throttle raising by 5s to avoid unnecessary requests for raising and clearing shortly after
                await asyncio.sleep(5)
            if action == AlertAction.RAISE and self.alert_key_to_last_clear_tstamp.get(key, 0) > self.alert_key_to_last_raise_tstamp.get(key, 0):
                logger.warning(f"Avoiding raising alert key [{key}] bc cleared shortly after")
                self.tasks_currently_running -= 1
                return None
            try:
                logger.info(f"Attempting {action} {f'[{key}] ' if key is not None else ''}request")
                response = await self.get_loop().run_in_executor(self.executor, api_call)
                self.tasks_currently_running -= 1
                return response
            except Exception as e:
                logger.error(f"Failed to make request {api_call}: {e}")
                self.tasks_currently_running -= 1
                return None
        self.tasks_currently_running -= 1
        return None

    async def _bootstrap(self):
        """
        Fetch open active alerts
        """
        if is_opsgenie_alerting_disabled():
            return
        bootstrap_query = "status=open AND details.key:StatArb"
        while True:
            current_page = 0
            page_size = 100
            api_call = functools.partial(self.alert_api.list_alerts, limit=page_size, offset=current_page * page_size, query=bootstrap_query)
            list_response = await self.execute_call_with_retry(api_call, AlertAction.BOOTSTRAP)

            if list_response and list_response.data:
                logger.info(f"Refreshing bookstrap response {list_response.data}")
                self.active_alert_keys.clear()

            while list_response and list_response.data:
                logger.info(f"Processing bootstrap response of length {len(list_response.data)} for page {current_page}")
                for alert in list_response.data:
                    self.active_alert_keys.add(alert.message)
                    self.active_alert_key_to_update_tstamp[alert.message] = max(alert.last_occurred_at.timestamp(), alert.updated_at.timestamp())

                if list_response.paging.next:
                    current_page += 1
                    api_call = functools.partial(self.alert_api.list_alerts, limit=page_size, offset=current_page * page_size, query=bootstrap_query)
                    list_response = await self.execute_call_with_retry(api_call, AlertAction.BOOTSTRAP)
                else:
                    list_response = None
                    self.bootstrapped = True
            if list_response and not list_response.data:
                self.bootstrapped = True
                logger.info("Done bootstrap")

            try:
                if not self.bootstrapped:
                    await asyncio.sleep(SECONDS_IN_MINUTE)
                elif self.disable_forever_bootstrap:
                    break
                else:
                    await asyncio.sleep(30 * SECONDS_IN_MINUTE + random.uniform(-3 * SECONDS_IN_MINUTE, 3 * SECONDS_IN_MINUTE))
            except asyncio.CancelledError:
                break  # Exit the loop if the task is cancelled

            except Exception as e:
                logger.error(f"bootstrap error {e}")

    async def update_alert_dict(self, update_from_bootstrap: bool = False):
        last_alert_dict = await self.load_alert_dict()
        logger.info(f"Getting latest alerts from local {last_alert_dict}")
        if update_from_bootstrap:
            await self.wait_for_initial_bootstrap()
            data = {'last_update': time.time(), 'data': {}}
        else:
            data = {'last_update': last_alert_dict.get('last_update', 0), 'data': {}}

        for key in self.active_alert_keys:
            timestamp = self.active_alert_key_to_update_tstamp[key]
            data['data'][key] = timestamp
        logger.info(f"Updating latest alerts from api {update_from_bootstrap=} {data}")
        async with asyncio.Lock():
            with open(self.alerts_filename, 'w') as f:
                json.dump(data, f)

    async def load_alert_dict(self) -> dict:
        async with asyncio.Lock():
            if not os.path.exists(self.alerts_filename):
                logger.info(f"File {self.alerts_filename} not found. Returning an empty dictionary.")
                return {}
            with open(self.alerts_filename, 'r') as f:
                return json.load(f)

    async def bootstrap_from_local(self):
        """
        Fetch open active alerts from local
        """
        logger.info("Bootstrap from local")
        if is_opsgenie_alerting_disabled():
            return

        while True:
            try:
                active_alerts = await self.load_alert_dict()
                logger.info(f"Getting active alerts from local {active_alerts}")
                self.active_alert_keys.clear()
                # sync alerts if local copy was 15 mins ago
                if time.time() - active_alerts.get('last_update', 0) < 15 * SECONDS_IN_MINUTE:
                    for alert_key in active_alerts.get('data', {}).keys():
                        self.active_alert_keys.add(alert_key)
                        self.active_alert_key_to_update_tstamp[alert_key] = active_alerts.get('data', {}).get(alert_key)
                    self.bootstrapped = True
                else:
                    logger.info("Alert dict too stale, try bootstrap from api")
                    await self._bootstrap()
                    await self.update_alert_dict(update_from_bootstrap=True)
                    await asyncio.sleep(1)
                    return
            except Exception as e:
                # If failed to fetch active alerts from local, revert back to opsgenie api and exit this function
                logger.error(f"Failed to bootstrap active alerts from mongoDB: {e}")
                await self._bootstrap()
                return

            if self.disable_forever_bootstrap:
                break
            else:
                await asyncio.sleep(30)

    async def wait_for_initial_bootstrap(self):
        """
        Wait until bootstrapping prior to alert action
        """
        if not self.bootstrapped:
            async with self._bootstrap_lock:
                start_ts = time.time()
                while not self.bootstrapped and time.time() - start_ts < SECONDS_IN_MINUTE / 2:
                    logger.info(f"{self} waiting for bootstrap to finish")
                    await asyncio.sleep(2)

                if not self.bootstrapped:
                    logger.error("Bootstrap timed out so will proceed without bootstrapped alerts")
                    self.bootstrapped = True

    async def raise_alert(self, key: str, priority: str, description: str, append_to_description: bool = False):
        """
        Create or update alert description if already exists
        Currently we ignore append_to_description bc of opsgenie smaller description char limit
        """
        logger.info(f"Try raise alert {key} with priority {priority}")
        if time.time() - self.alert_key_to_last_raise_tstamp.get(key, 0) < MIN_WAIT_RAISE_ALERT:
            logger.info(f"not raising alert [{key}] as it has been updated too recently")
            return
        self.alert_key_to_last_raise_tstamp[key] = time.time()
        await self.wait_for_initial_bootstrap()

        if any(pattern in key for pattern in self.disabled_alert_keys):
            logger.info(f"Alert [{key}] is disabled")
            return
        self.alert_key_to_last_clear_tstamp.pop(key, None)

        if key not in self.active_alert_keys:
            logger.info("Key not in current active_alert_keys, so raise new alert")
            body = opsgenie_sdk.CreateAlertPayload(
                message = key,
                alias = alert_key_hash(key),
                description = description,
                details = self.details,
                priority = PRIORITY_TO_OPSGENIE_PRI.get(priority),
            )
            api_call = functools.partial(self.alert_api.create_alert, create_alert_payload=body)
        else:
            logger.info("Key already in current active_alert_keys, so update existing alert")
            if append_to_description:
                body = opsgenie_sdk.AddNoteToAlertPayload(note=description)
                api_call = functools.partial(self.alert_api.add_note, identifier=alert_key_hash(key), add_note_to_alert_payload=body, identifier_type="alias")
                response = await self.execute_call_with_retry(api_call, AlertAction.ADD_NOTE, key)

            if isinstance(description, str):
                description += f"\n{self.details_str}"
            body = opsgenie_sdk.UpdateAlertDescriptionPayload(description=description)
            api_call = functools.partial(
                self.alert_api.update_alert_description, update_alert_description_payload=body, identifier=alert_key_hash(key), identifier_type="alias"
            )
            if self.alert_key_to_last_priority.get(key) != priority:
                body_priority_update = opsgenie_sdk.UpdateAlertPriorityPayload(priority=PRIORITY_TO_OPSGENIE_PRI.get(priority))
                api_call_priority_update = functools.partial(
                    self.alert_api.update_alert_priority,
                    update_alert_priority_payload = body_priority_update,
                    identifier = alert_key_hash(key),
                    identifier_type = "alias",
                )
                response = await self.execute_call_with_retry(api_call_priority_update, AlertAction.UPDATE_PRIORITY, key)
        self.active_alert_keys.add(key)
        self.alert_key_to_last_priority[key] = priority
        self.active_alert_key_to_update_tstamp[key] = time.time()
        response = await self.execute_call_with_retry(api_call, AlertAction.RAISE if key not in self.active_alert_keys else AlertAction.UPDATE_DESCRIPTION, key)
        logger.info(f"Response for raising alert [{key}]: {response}")

        await self.update_alert_dict()

    async def clear_alert(self, key: str):
        """ "
        Close alert if exists in active alerts
        """
        logger.info(f"Try clear alert {key}")
        if time.time() - self.alert_key_to_last_clear_tstamp.get(key, 0) < MIN_WAIT_RAISE_ALERT:
            logger.info(f"alert [{key}] has been cleared too recently without reraise")
            return
        self.alert_key_to_last_clear_tstamp[key] = time.time()
        await self.wait_for_initial_bootstrap()
        if key in self.active_alert_keys or (
            (not self.bootstrapped) and time.time() - self.alert_key_to_last_close_req.get(key, 0) > 10 * SECONDS_IN_MINUTE
        ):
            body = opsgenie_sdk.CloseAlertPayload()
            api_call = functools.partial(self.alert_api.close_alert, identifier=alert_key_hash(key), close_alert_payload=body, identifier_type="alias")
            self.alert_key_to_last_close_req[key] = time.time()
            close_response = await self.execute_call_with_retry(api_call, AlertAction.CLEAR, key)
            logger.info(f"Response for clearing alert [{key}]: {close_response}")
            if close_response is not None:
                self.active_alert_keys.discard(key)
                # We want to reraise without delay after clearing
                self.alert_key_to_last_raise_tstamp.pop(key, None)
                self.active_alert_key_to_update_tstamp.pop(key, None)
                await self.update_alert_dict()
        else:
            logger.info(f"No existing alert for key [{key}], so not clearing")

    async def shutdown(self, time_to_wait: int = 0):
        """
        Process shutdown once all alerting has completed or sufficient time elapsed
        """
        logger.info("got shutdown call")
        shutdown_start_ts = time.time()

        if time_to_wait:
            self.bootstrapped = False
            while time.time() - shutdown_start_ts <= time_to_wait and not self.bootstrapped:
                logger.info("Waiting for bootstrap before shutting down")
                await asyncio.sleep(0.25)

        if not self.bootstrapped:
            logger.info("Assuming bootstrapped for shutdown")
            self.bootstrapped = True

        if self._bootstrap_task and not self._bootstrap_task.done():
            self._bootstrap_task.cancel()

        await asyncio.sleep(2)
        while self.tasks_currently_running > 0 and time.time() - shutdown_start_ts < SECONDS_IN_MINUTE:
            logger.info("Awaiting task completion before shutting down")
            await asyncio.sleep(5)
        if self.tasks_currently_running > 0:
            logger.error(f"shutting down impolitely with {self.tasks_currently_running} tasks running")

        # Shutdown the thread pool executor
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)

        logger.info("OpsGenie client has been shut down")

    async def escalate_alert_to_critical(self, alert_key_substr: str):
        """
        Escalate the given alert (identified by a substring of its name) to critical (P1) and assign it to the provided user.
        """
        try:
            await self._bootstrap()
            await self.wait_for_initial_bootstrap()
            # Find the alert that matches the given substring
            alert_key = next((alert for alert in self.active_alert_keys if alert_key_substr.lower() in alert.lower()), None)
            if not alert_key:
                logger.error(f"Alert containing '{alert_key_substr}' not found in {self.active_alert_keys}")
                return
            await asyncio.sleep(1)
            # Update the alert priority to P1 (critical)
            priority_payload = opsgenie_sdk.UpdateAlertPriorityPayload(priority=PRIORITY_TO_OPSGENIE_PRI.get(CRITICAL))
            update_priority_response = await self.execute_call_with_retry(
                functools.partial(
                    self.alert_api.update_alert_priority,
                    identifier = alert_key_hash(alert_key),
                    identifier_type = "alias",
                    update_alert_priority_payload = priority_payload,
                ),
                AlertAction.UPDATE_PRIORITY,
                key = alert_key,
            )
            logger.info(f"Update priority response: {update_priority_response}")

        except Exception as e:
            logger.exception(f"Failed to escalate alert {alert_key_substr} to critical: {e}")

    def __str__(self):
        return "OpsGenie"


def raise_alert(key: str, priority: str, description: str, append_to_description: bool = False):
    logger.info("Raise alert called")
    if not is_opsgenie_alerting_disabled():
        new_key = '[StatArb] ' + key
        ops_singleton = OpsGenie.singleton()
        loop = ops_singleton.get_loop()
        bootstrap_task = loop.create_task(ops_singleton.bootstrap_from_local())
        alert_task = loop.create_task(ops_singleton.raise_alert(new_key, priority, description, append_to_description))
        try:
            loop.run_until_complete(asyncio.wait_for(asyncio.gather(bootstrap_task, alert_task), timeout=30))
        except asyncio.TimeoutError:
            logger.error(f"Timeout while raising alert '{key}' - alert may not have been sent")
        except Exception as e:
            logger.error(f"Error raising alert '{key}': {e}")


def clear_alert(key: str):
    logger.info("Clear alert called")
    if not is_opsgenie_alerting_disabled():
        new_key = '[StatArb] ' + key
        ops_singleton = OpsGenie.singleton()
        loop = ops_singleton.get_loop()
        bootstrap_task = loop.create_task(ops_singleton.bootstrap_from_local())
        clear_task = loop.create_task(ops_singleton.clear_alert(new_key))
        try:
            loop.run_until_complete(asyncio.wait_for(asyncio.gather(bootstrap_task, clear_task), timeout=30))
        except asyncio.TimeoutError:
            logger.error(f"Timeout while clearing alert '{key}' - alert may not have been cleared")
        except Exception as e:
            logger.error(f"Error clearing alert '{key}': {e}")


def escalate_alert(alert_key_substr: str):
    if not is_opsgenie_alerting_disabled():
        ops_singleton = OpsGenie.singleton()
        loop = ops_singleton.get_loop()
        loop.create_task(ops_singleton.bootstrap_from_local())
        loop.create_task(ops_singleton.escalate_alert_to_critical(alert_key_substr))
        loop.run_until_complete(asyncio.wait_for(asyncio.gather(*asyncio.all_tasks(loop)), timeout=15))


def shutdown_opsgenie(time_to_wait: int = 0):
    asyncio.ensure_future(OpsGenie.singleton().shutdown(time_to_wait=time_to_wait))


# On-call monitoring utilities - configure via environment
STATARB_SCHEDULE_ID = os.environ.get('OPSGENIE_SCHEDULE_ID', '')
STATARB_SCHEDULE_NAME = os.environ.get('OPSGENIE_SCHEDULE_NAME', 'default_schedule')
STATE_FILE = LOG_DIR + "/oncall_state.json"


def extract_name_from_email(email: str) -> str:
    """
    Extract a readable name from an email address.

    Args:
        email: Email address (e.g., "john.doe@example.com")

    Returns:
        str: Capitalized name (e.g., "John Doe")

    Example:
        >>> extract_name_from_email('john.doe@example.com')
        'John Doe'
        >>> extract_name_from_email('jane_smith@company.org')
        'Jane Smith'
    """
    if '@' in email:
        local_part = email.split('@')[0]
    else:
        local_part = email
    name = local_part.replace('.', ' ').replace('_', ' ').replace('-', ' ')
    name = ' '.join(word.capitalize() for word in name.split())
    return name


def get_current_oncall(api_key: str, schedule_id: str) -> dict:
    """
    Fetch the current on-call person from Opsgenie.

    Args:
        api_key: Opsgenie API key used for authentication
        schedule_id: The ID of the Opsgenie schedule to query

    Returns:
        dict: A dictionary with keys:
            - 'name' (str): The readable name of the on-call person
            - 'username' (str): The username or email of the on-call person

    Raises:
        requests.exceptions.HTTPError: If the HTTP request to Opsgenie fails
        requests.exceptions.RequestException: For other network-related errors
    """
    url = f"https://api.opsgenie.com/v2/schedules/{schedule_id}/on-calls"
    headers = {"Authorization": f"GenieKey {api_key}"}
    params = {"scheduleIdentifierType": "id", "flat": "true"}

    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    logger.debug(f"API response: {json.dumps(data, indent=2)}")

    on_calls = data.get('data', {}).get('onCallRecipients', [])
    if not on_calls:
        raise log_and_raise("No on-call recipients found in Opsgenie schedule")

    primary_oncall = on_calls[0]
    if isinstance(primary_oncall, str):
        full_name = extract_name_from_email(primary_oncall)
        return {"name": full_name, "username": primary_oncall}

    username = primary_oncall.get('name', 'Unknown')
    full_name = extract_name_from_email(username)
    return {"name": full_name, "username": username}


def load_oncall_state() -> dict:
    """Load the previous on-call state from file."""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load previous state: {e}")
        return {}


def save_oncall_state(oncall_info: dict) -> None:
    """Save the current on-call state to file."""
    state = {
        "name": oncall_info["name"],
        "username": oncall_info["username"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schedule": STATARB_SCHEDULE_NAME
    }
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved state: {state}")
    except Exception as e:
        logger.error(f"Could not save state: {e}")
