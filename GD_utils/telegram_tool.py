# $ pip install python-telegram-bot
import telegram

def send_telegram(msg, bot_token, chat_id, printing=True):
    bot = telegram.Bot(token=bot_token)
    bot.send_message(chat_id, msg)
    if printing:
        print(f'텔레그램 송출 완료: {msg}')