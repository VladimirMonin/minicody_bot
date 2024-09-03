"""
Модуль Telegram-бота для поддержки студентов в группах.
Бота можно добавить в телеграмм чаты, где студенты, смогут обращатся к нему по нику, получая ответы на свои вопросы.
Бот поддерживает асинхронную работу, ограничивает количество обращений студентов в сутки и обращений к OpenAI API.
Контекст общения сохраняется в JSON формате.
Модуль использует принципы единой ответственности (SRP) и избегает повторений кода (DRY).
"""

import json
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List
from telegram import Update
from telegram.ext import Application, MessageHandler, filters
from openai import AsyncOpenAI
from telegram.constants import ParseMode
from settings import ALLOWED_CHATS, MAX_MESSAGES_PER_DAY, CONTEXT_EXPIRATION_MINUTES, CONTEXT_MESSAGE_LIMIT, JSON_LOG_FILE, OPEN_AI_API_KEY, BOT_TOKEN, BOT_ROLE

# Настройка логгера
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные
chat_logs: Dict[int, Dict[int, List[Dict[str, str]]]] = {}
message_counters: Dict[int, Dict[int, int]] = {}

# Инициализация клиента OpenAI
openai_client = AsyncOpenAI(api_key=OPEN_AI_API_KEY)

async def update_bot_context(chat_id: int, user_id: int, message: str) -> None:
    """Обновляет контекст пользователя новым сообщением бота."""
    timestamp = time.time()
    if chat_id not in chat_logs:
        chat_logs[chat_id] = {}
    if user_id not in chat_logs[chat_id]:
        chat_logs[chat_id][user_id] = []
    chat_logs[chat_id][user_id].append({
        "timestamp": timestamp,
        "message": message,
        "human_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role": "assistant"
    })
    await save_chat_logs()

async def load_chat_logs() -> None:
    """Загружает логи общения из JSON файла."""
    global chat_logs
    try:
        with open(JSON_LOG_FILE, "r", encoding="utf-8") as file:
            chat_logs = json.load(file)
    except FileNotFoundError:
        logger.info("JSON файл с логами не найден. Создаем новый.")
        chat_logs = {}

async def save_chat_logs() -> None:
    """Сохраняет текущие логи общения в JSON файл."""
    with open(JSON_LOG_FILE, "w", encoding="utf-8") as file:
        json.dump(chat_logs, file, indent=4, ensure_ascii=False)

async def get_user_context(chat_id: int, user_id: int) -> List[Dict[str, str]]:
    """Возвращает контекст последних сообщений пользователя и ответов бота, если они не устарели."""
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
    try:
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id

        logger.info(f"Получено сообщение из чата {chat_id} от пользователя {user_id}")

        if chat_id not in ALLOWED_CHATS:
            logger.info(f"Сообщение из неразрешенного чата: {chat_id}")
            return  # Игнорируем сообщения из неразрешенных чатов

        # Проверяем, является ли чат групповым
        chat_type = update.effective_chat.type
        logger.info(f"Тип чата: {chat_type}")
        if chat_type == "private":
            logger.info(f"Сообщение из личного чата: {chat_id}")
            return  # Игнорируем сообщения из личных чатов

        # Проверяем, является ли пользователь администратором группы
        user_status = (await update.effective_chat.get_member(user_id)).status
        is_admin = user_status in ["creator", "administrator"]

        # Проверяем наличие текста в сообщении
        if update.message.text is None:
            logger.info(f"Сообщение не содержит текста: {update.message}")
            return

        message_text = update.message.text
        reply_to_message = update.message.reply_to_message
        bot_username = context.bot.username

        quoted_text = None
        if reply_to_message:
            if reply_to_message.from_user.is_bot:
                # Если сообщение является ответом на сообщение бота
                quoted_text = reply_to_message.text
                if message_text.startswith(quoted_text):
                    # Если цитируется все сообщение бота
                    message_text = message_text[len(quoted_text):].strip()
                else:
                    # Если цитируется только часть сообщения бота
                    message_text = message_text.strip()
                    quoted_text = message_text
            else:
                # Если сообщение является ответом на сообщение другого пользователя
                if f"@{bot_username}" in message_text:
                    # Если бот упомянут в сообщении
                    quoted_text = reply_to_message.text
                    message_text = message_text.replace(f"@{bot_username}", "").strip()
                else:
                    # Если бот не упомянут, игнорируем сообщение
                    logger.info(f"Сообщение является ответом на сообщение другого пользователя без упоминания бота: {message_text}")
                    return
        else:
            # Если сообщение не является ответом на другое сообщение
            if not message_text.startswith(f"@{bot_username}"):
                logger.info(f"Сообщение не адресовано боту: {message_text}")
                return
            message_text = message_text.replace(f"@{bot_username}", "").strip()

        logger.info(f"Обработка сообщения: {message_text}")

        if not is_admin and message_counters.get(chat_id, {}).get(user_id, 0) > MAX_MESSAGES_PER_DAY:
            await update.message.reply_text("Вы превысили лимит сообщений на сегодня.")
            logger.info(f"Превышен лимит сообщений для пользователя: {user_id}")
            return

        # Обновляем и получаем контекст
        await log_message(chat_id, user_id, message_text)
        context_messages = await get_user_context(chat_id, user_id)

        # Запрос к OpenAI API
        response = await openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": BOT_ROLE},
                *[{"role": msg.get("role", "user"), "content": msg["message"]} for msg in context_messages],
                {"role": "user", "content": quoted_text if quoted_text else message_text}
            ],
            model="gpt-4o-mini"
        )

        reply_text = response.choices[0].message.content

        # Разбиваем длинный ответ на несколько сообщений
        max_length = 4096
        reply_chunks = [reply_text[i:i+max_length] for i in range(0, len(reply_text), max_length)]
        for chunk in reply_chunks:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

        await update_bot_context(chat_id, user_id, reply_text)  # Сохраняем ответ бота в контекст
        logger.info(f"Отправлен ответ пользователю {user_id}: {reply_text}")
    except Exception as e:
        logger.exception(f"Ошибка при обработке сообщения: {e}")




async def main() -> None:
    """Основная функция для запуска бота."""
    try:
        await load_chat_logs()

        application = Application.builder().token(BOT_TOKEN).build()

        # Обработчики сообщений
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        # Запуск бота
        logger.info("Запуск бота...")
        await application.initialize()
        await application.start()
        await application.updater.start_polling()
    except Exception as e:
        logger.exception(f"Ошибка при запуске бота: {e}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()



