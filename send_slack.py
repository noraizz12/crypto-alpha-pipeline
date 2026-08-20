#!/usr/bin/env python
"""Send a message to a Slack channel via webhook."""
import argparse
import sys

from slack_sdk.webhook.client import WebhookClient

from lib.util.slack import (
    SLACK_WEBHOOK,
    SLACK_JOBS_WEBHOOK,
    SLACK_PROD_JOBS,
    SLACK_PNL_WEBHOOK,
    SLACK_ONCALL_WEBHOOK,
)

CHANNEL_WEBHOOKS = {
    'default': SLACK_WEBHOOK,
    'jobs': SLACK_JOBS_WEBHOOK,
    'prod': SLACK_PROD_JOBS,
    'pnl': SLACK_PNL_WEBHOOK,
    'oncall': SLACK_ONCALL_WEBHOOK,
}


def main():
    """Parse arguments and send message to Slack."""
    parser = argparse.ArgumentParser(description='Send a message to a Slack channel')
    parser.add_argument('--channel', '-c', type=str, default='default',
                        choices=list(CHANNEL_WEBHOOKS.keys()),
                        help=f'Slack channel to send to: {list(CHANNEL_WEBHOOKS.keys())}')
    parser.add_argument('--message', '-m', type=str, required=True,
                        help='Message to send')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Debug mode - print message instead of sending')
    args = parser.parse_args()

    webhook_url = CHANNEL_WEBHOOKS[args.channel]

    if args.debug:
        print(f"Channel: {args.channel}")
        print(f"Message: {args.message}")
        print(f"Webhook: {webhook_url[:50]}...")
        return 0

    client = WebhookClient(webhook_url)
    response = client.send(text=args.message)

    if response.status_code == 200:
        print(f"Message sent to '{args.channel}' channel")
        return 0

    print(f"Failed to send message: {response.status_code} - {response.body}")
    return 1


if __name__ == '__main__':
    sys.exit(main())
