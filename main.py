"""
Модуль Telegram-бота для поддержки студентов в группах.
Бот поддерживает асинхронную работу, ограничивает количество обращений студентов в сутки и обращений к OpenAI API.
Контекст общения сохраняется в JSON формате.
Модуль использует принципы единой ответственности (SRP) и избегает повторений кода (DRY).
"""

import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from openai import AsyncOpenAI

from settings import ALLOWED_CHATS, MAX_MESSAGES_PER_DAY, CONTEXT_EXPIRATION_MINUTES, CONTEXT_MESSAGE_LIMIT, OPENAI_API_LIMIT_PER_SECOND, JSON_LOG_FILE, OPEN_AI_API_KEY, BOT_TOKEN

# Глобальные переменные
chat_logs: Dict[int, Dict[int, List[Dict[str, str]]]] = {}
message_counters: Dict[int, Dict[int, int]] = {}

# Инициализация клиента OpenAI
openai_client = AsyncOpenAI(api_key=OPEN_AI_API_KEY)

async def load_chat_logs() -> None:
    """Загружает логи общения из JSON файла."""
    global chat_logs
    try:
        with open(JSON_LOG_FILE, "r", encoding="utf-8") as file:
            chat_logs = json.load(file)
    except FileNotFoundError:
        chat_logs = {}


async def save_chat_logs() -> None:
    """Сохраняет текущие логи общения в JSON файл."""
    with open(JSON_LOG_FILE, "w", encoding="utf-8") as file:
        json.dump(chat_logs, file, indent=4, ensure_ascii=False)


async def get_user_context(chat_id: int, user_id: int) -> List[Dict[str, str]]:
    """Возвращает контекст последних сообщений пользователя, если они не устарели."""
    if chat_id in chat_logs and user_id in chat_logs[chat_id]:
        context = []
        for msg in chat_logs[chat_id][user_id][-CONTEXT_MESSAGE_LIMIT:]:
            if time.time() - msg["timestamp"] < CONTEXT_EXPIRATION_MINUTES * 60:
                context.append(msg)
        return context
    return []


async def update_user_context(chat_id: int, user_id: int, message: str) -> None:
    """Обновляет контекст пользователя новым сообщением."""
    timestamp = time.time()
    if chat_id not in chat_logs:
        chat_logs[chat_id] = {}
    if user_id not in chat_logs[chat_id]:
        chat_logs[chat_id][user_id] = []
    chat_logs[chat_id][user_id].append({
        "timestamp": timestamp,
        "message": message,
        "human_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    if len(chat_logs[chat_id][user_id]) > CONTEXT_MESSAGE_LIMIT:
        chat_logs[chat_id][user_id].pop(0)
    await save_chat_logs()


async def log_message(chat_id: int, user_id: int, message: str) -> None:
    """Логирует новое сообщение студента."""
    if chat_id not in message_counters:
        message_counters[chat_id] = {}
    if user_id not in message_counters[chat_id]:
        message_counters[chat_id][user_id] = 0
    message_counters[chat_id][user_id] += 1

    if message_counters[chat_id][user_id] <= MAX_MESSAGES_PER_DAY:
        await update_user_context(chat_id, user_id, message)


async def handle_message(update: Update, context) -> None:
    """Обрабатывает входящие сообщения от пользователей."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    if chat_id not in ALLOWED_CHATS:
        return  # Игнорируем сообщения из неразрешенных чатов

    if message_counters.get(chat_id, {}).get(user_id, 0) > MAX_MESSAGES_PER_DAY:
        await update.message.reply_text("Вы превысили лимит сообщений на сегодня.")
        return

    message_text = update.message.text

    # Обновляем и получаем контекст
    await log_message(chat_id, user_id, message_text)
    context_messages = await get_user_context(chat_id, user_id)

    # Запрос к OpenAI API
    response = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Ты бот-поддержки, помогай студентам, но не давай готовые решения."},
            *[{"role": "user", "content": msg["message"]} for msg in context_messages],
            {"role": "user", "content": message_text}
        ],
        model="gpt-4"
    )

    reply_text = response["choices"][0]["message"]["content"]
    await update.message.reply_text(reply_text)


async def main() -> None:
    """Основная функция для запуска бота."""
    await load_chat_logs()

    application = Application.builder().token(BOT_TOKEN).build()

    # Обработчики сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    await application.run_polling()


if __name__ == "__main__":
    asyncio.run(main())
