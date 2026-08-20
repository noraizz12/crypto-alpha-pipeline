import os
import getpass
import socket
import sys
import time
from datetime import datetime, timezone
import traceback
from typing import Optional, Tuple
from types import TracebackType

from slack_sdk.webhook.async_client import AsyncWebhookClient
from slack_sdk.webhook.client import WebhookClient

# Slack webhooks loaded from environment variables for security
# Set these in .env or environment before running
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")
SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK", "")
SLACK_JOBS_WEBHOOK = os.environ.get("SLACK_JOBS_WEBHOOK", "")
SLACK_PROD_JOBS = os.environ.get("SLACK_PROD_JOBS_WEBHOOK", "")
SLACK_PNL_WEBHOOK = os.environ.get("SLACK_PNL_WEBHOOK", "")
SLACK_ONCALL_WEBHOOK = os.environ.get("SLACK_ONCALL_WEBHOOK", "")

# Default Slack member ID for notifications (override via environment)
DEFAULT_SLACK_MEMBER_ID = os.environ.get("SLACK_MEMBER_ID", "")


async def send_slack_async(webhook: AsyncWebhookClient, msg: str):
    try:
        await webhook.send(text=msg)
    except TimeoutError:
        pass


class JobMonitor:
    """Context manager to monitor a job and send notifications"""
    def __init__(self, job_name: str, slack_channel: Optional[str], debug: bool = False, quiet: bool = False) -> None:
        self.job_name = job_name
        self.start_time: Optional[float] = None
        self.hostname = socket.gethostname()
        self.username = getpass.getuser()
        self.quiet = quiet
        slack_hook, slack_user = self.get_slack_channel_hook(slack_channel)
        self.slack_user_call = f'<@{slack_user}>' if slack_user is not None else ''
        self.slack_client = WebhookClient(slack_hook) if slack_hook is not None and not debug and not quiet else None
        self.script_name: Optional[str] = None
        self.script_args: Optional[str] = None

    def get_slack_channel_hook(self, slack_channel: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        slack_hook = SLACK_PROD_JOBS if 'prod' in self.hostname else SLACK_JOBS_WEBHOOK
        slack_user = None
        if slack_channel is not None and DEFAULT_SLACK_MEMBER_ID:
            slack_user = DEFAULT_SLACK_MEMBER_ID
        return slack_hook, slack_user

    def __enter__(self) -> 'JobMonitor':
        self.start_time = time.time()
        self.notify_slack("Started", self.start_time)
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[TracebackType]) -> bool:
        if exc_type is None or (exc_type is SystemExit and hasattr(exc_val, 'code') and exc_val.code == 0):
            # Job succeeded
            self.notify_slack("Success", self.start_time, time.time())
        else:
            # Job failed
            error = str(exc_val) + "\n" + "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            self.notify_slack("Failed", self.start_time, time.time(), error)

        # Don't suppress the exception
        return False

    def notify_slack(self, status: str, start_time: float, end_time: Optional[float] = None, error: Optional[str] = None) -> None:
        """Send a notification to Slack"""
        start_timestamp = datetime.fromtimestamp(start_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        if status == "Started":
            message = f"🚀 *Job Started:* `{self.job_name}`{self.slack_user_call}\n*Host:* `{self.username}@{self.hostname}`\n*Started at:* {start_timestamp}"
            self.script_name = os.path.basename(sys.argv[0])
            self.script_args = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "None"
            message += f"\n*Script:* {self.script_name}\n*Arguments:* {self.script_args}"
        else:
            duration = end_time - start_time if end_time else 0
            end_timestamp = datetime.fromtimestamp(end_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            if status == "Success":
                message = f"✅ *Job Complete:* `{self.job_name}`{self.slack_user_call}\n*Host:* `{self.username}@{self.hostname}`"
            else:
                message = f"❌ *Job Failed:* `{self.job_name}`{self.slack_user_call}\n*Host:* `{self.username}@{self.hostname}`"

            message += f"\n*Script:* {self.script_name}\n*Arguments:* {self.script_args}"
            message += f"\n*Status:* {status}"
            message += f"\n*Started at:* {start_timestamp}"
            message += f"\n*Completed at:* {end_timestamp}"
            message += f"\n*Duration:* {duration:.2f} seconds"

            if error:
                message += f"\n*Error:*\n```{error[:1000]}```"
        if self.slack_client is not None:
            self.slack_client.send(text=message)
