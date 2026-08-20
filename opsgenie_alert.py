import sys
import argparse

from lib.util.opsgenie import AlertAction, HIGH, raise_alert, clear_alert


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Call Opsgenie')
    parser.add_argument('-f', '--func', help='opsgenie function', default=AlertAction.RAISE)
    parser.add_argument('-k', '--key', help='opsgenie key', default='test opsgenie alert')
    parser.add_argument('-p', '--priority', help='opsgenie priority', default=HIGH)
    parser.add_argument('-d', '--description', help='opsgenie description', default='description of opsgenie alert')
    parser.add_argument('-a', '--append', help='opsgenie append_to_description', default=False, action="store_true")

    args = vars(parser.parse_args())
    if args['func'] == AlertAction.RAISE:
        raise_alert(args['key'], args['priority'], args['description'], args['append'])

    elif args['func'] == AlertAction.CLEAR:
        clear_alert(args['key'])
    else:
        print(f"Unsupported function: {args['func']}")
        sys.exit(1)