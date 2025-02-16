from typing import Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from sdl_orchestration import sdl_config
from sdl_orchestration.notification.templates.message_factory import \
    MessageFactory

slack_client = WebClient(token=sdl_config.slack_bot_token)


class NotificationClient:
    """
    This class is used to send notification to the Slack channel.
    """
    DIVIDER_BLOCK = {"type": "divider"}

    def __init__(self):
        self.channel = sdl_config.notification_config["channel"]
        self.MENTION_FLAG = "<!channel>"
        self.message_factory = MessageFactory()

    def send_notification(self,
                          message_type: str,
                          mention_all: bool = False,
                          *args, **kwargs):
        """
        This method sends the notification to the Slack channel.

        Args:
            message_type (str): The type of the message.
            mention_all (bool): Flag to mention all.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        message_payload = self._parse_message(message_type=message_type,
                                              mention_all=mention_all, *args,
                                              **kwargs)
        self._send_message(message_payload)

    def _parse_message(self, message_type: str, mention_all: bool = False,
                       *args, **kwargs):
        """
        This method parses the message based on the message type.
        It also adds mention all flag if needed.

        Args:
            message_type (str): The type of the message.
            mention_all (bool): Flag to mention all.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        # Select a template based on the message
        message = self.message_factory.create_message(message_type,
                                                      *args,
                                                      **kwargs).parse()

        # Add mention all flag if needed
        if mention_all:
            message = f"{self.MENTION_FLAG} {message}"

        # Construct the message payload
        blocks = [
            self._get_section_block(message),
            self.DIVIDER_BLOCK
        ]
        return {"channel": self.channel, "blocks": blocks,
                "text": "SDL Notification"}

    def _send_message(self, message_payload: dict):
        """
        This method sends the message to the Slack channel.
        """
        try:
            slack_client.chat_postMessage(**message_payload)
        except SlackApiError as e:
            assert not e.response["ok"]
            print(f"Got an error: {e.response['error']}")
            assert isinstance(e.response.status_code, int)
            print(f"Received a response status_code: {e.response.status_code}")
            raise e

    @staticmethod
    def _get_section_block(text: str):
        return {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text
            }
        }


notification_client = NotificationClient()
