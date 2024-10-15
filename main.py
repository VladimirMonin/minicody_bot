"""
Модуль Telegram-бота для поддержки студентов в группах.
Бот может быть добавлен в телеграмм чаты, где студенты смогут обращаться к нему по нику, получая ответы на свои вопросы.
Бот поддерживает асинхронную работу, ограничивает количество обращений студентов в сутки и обращений к OpenAI API.
Контекст общения сохраняется в JSON формате.
"""

import json
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from telegram import Update
from telegram.ext import Application, MessageHandler, filters
from openai import AsyncOpenAI
from telegram.constants import ParseMode
from settings import (
    ALLOWED_CHATS, MAX_MESSAGES_PER_DAY, CONTEXT_EXPIRATION_MINUTES,
    CONTEXT_MESSAGE_LIMIT, JSON_LOG_FILE, OPEN_AI_API_KEY, BOT_TOKEN,
    BOT_ROLE, MODEL
)

# Настройка логгера
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные
chat_logs: Dict[int, Dict[int, List[Dict[str, str]]]] = {}
message_counters: Dict[int, Dict[int, int]] = {}

# Инициализация клиента OpenAI
openai_client = AsyncOpenAI(api_key=OPEN_AI_API_KEY)

# ====================== УПРАВЛЕНИЕ КОНТЕКСТОМ ===============

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
    """
    Возвращает контекст последних сообщений пользователя и ответов бота, если они не устарели.
    
    :param chat_id: ID чата
    :param user_id: ID пользователя
    :return: Список словарей с контекстом сообщений
    """
    if chat_id in chat_logs and user_id in chat_logs[chat_id]:
        context = []
        for msg in chat_logs[chat_id][user_id][-CONTEXT_MESSAGE_LIMIT:]:
            if time.time() - msg["timestamp"] < CONTEXT_EXPIRATION_MINUTES * 60:
                context.append(msg)
        return context
    return []

async def update_user_context(chat_id: int, user_id: int, message: str, role: str = "user") -> None:
    """
    Обновляет контекст пользователя новым сообщением.
    
    :param chat_id: ID чата
    :param user_id: ID пользователя
    :param message: Текст сообщения
    :param role: Роль отправителя сообщения ("user" или "assistant")
    """
    timestamp = time.time()
    if chat_id not in chat_logs:
        chat_logs[chat_id] = {}
    if user_id not in chat_logs[chat_id]:
        chat_logs[chat_id][user_id] = []
    chat_logs[chat_id][user_id].append({
        "timestamp": timestamp,
        "message": message,
        "human_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role": role
    })
    await save_chat_logs()

# ====================== ОБРАБОТКА СООБЩЕНИЙ ===============

async def is_allowed_chat(chat_id: int) -> bool:
    """
    Проверяет, разрешен ли чат для работы бота.
    
    :param chat_id: ID чата
    :return: True, если чат разрешен, иначе False
    """
    return chat_id in ALLOWED_CHATS

async def is_group_chat(chat_type: str) -> bool:
    """
    Проверяет, является ли чат групповым.
    
    :param chat_type: Тип чата
    :return: True, если чат групповой, иначе False
    """
    return chat_type != "private"

async def is_admin(user_status: str) -> bool:
    """
    Проверяет, является ли пользователь администратором.
    
    :param user_status: Статус пользователя
    :return: True, если пользователь администратор, иначе False
    """
    return user_status in ["creator", "administrator"]

async def extract_message_text(message_text: str, reply_to_message: Optional[Update], bot_username: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Извлекает текст сообщения и цитируемый текст, если сообщение адресовано боту.
    
    :param message_text: Текст сообщения
    :param reply_to_message: Объект сообщения, на которое отвечают
    :param bot_username: Имя пользователя бота
    :return: Кортеж (цитируемый текст, текст сообщения) или (None, None), если сообщение не адресовано боту
    """
    quoted_text = None
    if reply_to_message:
        if reply_to_message.from_user.is_bot:
            quoted_text = reply_to_message.text
            if message_text.startswith(quoted_text):
                message_text = message_text[len(quoted_text):].strip()
            else:
                message_text = message_text.strip()
        elif f"@{bot_username}" in message_text:
            quoted_text = reply_to_message.text
            message_text = message_text.replace(f"@{bot_username}", "").strip()
        else:
            return None, None
    else:
        if not message_text.startswith(f"@{bot_username}"):
            return None, None
        message_text = message_text.replace(f"@{bot_username}", "").strip()
    
    logger.info(f"Извлечен текст сообщения: {message_text}")
    return quoted_text, message_text

async def check_message_limit(chat_id: int, user_id: int, is_admin: bool) -> bool:
    """
    Проверяет, не превышен ли лимит сообщений для пользователя.
    
    :param chat_id: ID чата
    :param user_id: ID пользователя
    :param is_admin: Флаг, указывающий, является ли пользователь администратором
    :return: True, если лимит не превышен, иначе False
    """
    if is_admin:
        return True
    return message_counters.get(chat_id, {}).get(user_id, 0) <= MAX_MESSAGES_PER_DAY

async def log_message(chat_id: int, user_id: int, message: str) -> None:
    """
    Логирует новое сообщение студента.
    
    :param chat_id: ID чата
    :param user_id: ID пользователя
    :param message: Текст сообщения
    """
    if chat_id not in message_counters:
        message_counters[chat_id] = {}
    if user_id not in message_counters[chat_id]:
        message_counters[chat_id][user_id] = 0
    message_counters[chat_id][user_id] += 1

    if message_counters[chat_id][user_id] <= MAX_MESSAGES_PER_DAY:
        await update_user_context(chat_id, user_id, message)

async def get_ai_response(context_messages: List[Dict[str, str]], message: str) -> str:
    """
    Получает ответ от OpenAI API.
    
    :param context_messages: Контекст предыдущих сообщений
    :param message: Текущее сообщение пользователя
    :return: Ответ от AI
    """
    response = await openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": BOT_ROLE},
            *[{"role": msg.get("role", "user"), "content": msg["message"]} for msg in context_messages],
            {"role": "user", "content": message}
        ],
        model=MODEL
    )
    return response.choices[0].message.content

async def send_response(update: Update, reply_text: str) -> None:
    """
    Отправляет ответ пользователю, разбивая длинные сообщения на части.
    
    :param update: Объект обновления Telegram
    :param reply_text: Текст ответа
    """
    max_length = 4096
    reply_chunks = [reply_text[i:i+max_length] for i in range(0, len(reply_text), max_length)]
    for chunk in reply_chunks:
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

async def handle_message(update: Update, context) -> None:
    """
    Обрабатывает входящие сообщения от пользователей.
    
    :param update: Объект обновления Telegram
    :param context: Контекст бота
    """
    try:
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        
        logger.info(f"Получено сообщение из чата {chat_id} от пользователя {user_id}")

        if not await is_allowed_chat(chat_id):
            logger.info(f"Сообщение из неразрешенного чата: {chat_id}")
            return

        chat_type = update.effective_chat.type
        if not await is_group_chat(chat_type):
            logger.info(f"Сообщение из личного чата: {chat_id}")
            return

        user_status = (await update.effective_chat.get_member(user_id)).status
        user_is_admin = await is_admin(user_status)

        if update.message.text is None:
            logger.info(f"Сообщение не содержит текста: {update.message}")
            return

        quoted_text, message_text = await extract_message_text(
            update.message.text,
            update.message.reply_to_message,
            context.bot.username
        )
        
        if message_text is None:
            logger.info(f"Сообщение не адресовано боту: {update.message.text}")
            return

        logger.info(f"Обработка сообщения: {message_text}")

        if not await check_message_limit(chat_id, user_id, user_is_admin):
            await update.message.reply_text("Вы превысили лимит сообщений на сегодня.")
            logger.info(f"Превышен лимит сообщений для пользователя: {user_id}")
            return

        await log_message(chat_id, user_id, message_text)
        context_messages = await get_user_context(chat_id, user_id)

        reply_text = await get_ai_response(context_messages, quoted_text if quoted_text else message_text)

        await send_response(update, reply_text)
        await update_user_context(chat_id, user_id, reply_text, "assistant")
        logger.info(f"Отправлен ответ пользователю {user_id}: {reply_text}")
    except Exception as e:
        logger.exception(f"Ошибка при обработке сообщения: {e}")

# ====================== УПРАВЛЕНИЕ БОТОМ ===============

async def reset_message_counters() -> None:
    """Сбрасывает счетчики сообщений каждые сутки."""
    while True:
        global message_counters
        message_counters = {}
        logger.info("Счетчики сообщений сброшены.")
        await asyncio.sleep(24 * 60 * 60)  # Ждем 24 часа

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

        # Запуск задачи для сброса счетчиков сообщений каждые сутки
        asyncio.create_task(reset_message_counters())
    except Exception as e:
        logger.exception(f"Ошибка при запуске бота: {e}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()