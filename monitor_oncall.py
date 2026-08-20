#!/usr/bin/env python3
"""Monitor Opsgenie on-call schedule and send Slack notifications when it changes."""

import argparse
import logging
import sys

import requests

from lib.util.aws import load_aws_secrets
from lib.util.opsgenie import (
    STATARB_SCHEDULE_ID,
    STATARB_SCHEDULE_NAME,
    get_current_oncall,
    load_oncall_state,
    save_oncall_state,
)
from lib.util.slack import SLACK_ONCALL_WEBHOOK

logger = logging.getLogger(__name__)


def send_slack_notification(oncall_info: dict, debug: bool = False) -> None:
    """Send Slack notification about on-call change."""
    name = oncall_info["name"]
    message = f":bell: {name} is now on call for support."

    if debug:
        print(f"[DEBUG] Would send to Slack: {message}")
        return

    try:
        payload = {"text": message}
        response = requests.post(
            SLACK_ONCALL_WEBHOOK,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        logger.info(f"Sent Slack notification: {message}")
    except Exception as e:
        logger.error(f"Failed to send Slack notification: {e}")


def main() -> int:
    """Main function to monitor on-call and send notifications."""
    parser = argparse.ArgumentParser(description="Monitor Opsgenie on-call schedule")
    parser.add_argument("--debug", action="store_true", help="Debug mode (don't send Slack)")
    parser.add_argument("--force", action="store_true", help="Force notification even if unchanged")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )

    try:
        api_key = load_aws_secrets(statarb_secretid='statarb/ops_genie_api_key')['ops_genie_api_key']

        logger.info(f"Checking on-call for schedule: {STATARB_SCHEDULE_NAME}")
        current_oncall = get_current_oncall(api_key, STATARB_SCHEDULE_ID)
        logger.info(f"Current on-call: {current_oncall['name']} ({current_oncall['username']})")

        previous_state = load_oncall_state()
        previous_name = previous_state.get('name')

        if args.force or previous_name != current_oncall['name']:
            if args.force:
                logger.info("Forcing notification (--force flag)")
            else:
                logger.info(f"On-call changed: {previous_name} -> {current_oncall['name']}")

            send_slack_notification(current_oncall, debug=args.debug)
            save_oncall_state(current_oncall)
        else:
            logger.info(f"On-call unchanged: {current_oncall['name']}")

    except Exception as e:
        logger.error(f"Error monitoring on-call: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
