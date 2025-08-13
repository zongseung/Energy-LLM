import requests
import json


def send_slack_message(webhook_url, message, username, icon_emoji=":robot_face"):
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "text": message,
        "username": username,
        "icon_emoji": icon_emoji
    }

    response = requests.post(webhook_url, headers=headers, data=json.dumps(payload))

    if response.status_code != 200:
        payload = {
        "text": "slack 알림 에러 발생",
        "username": username,
        "icon_emoji": icon_emoji
        }
        response = requests.post(webhook_url, headers=headers, data=json.dumps(payload))
        raise Exception(f"슬랙 알림 에러 발생 : {response.status_code}, 응답 텍스트 : \n{response.text}")


if __name__ == "__main__":
    webhook_url = "" # https://hooks.slack.com/services/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
    message = "Web UI 자동화 테스트가 시작 되었습니다. :web:"
    send_slack_message(webhook_url, message)