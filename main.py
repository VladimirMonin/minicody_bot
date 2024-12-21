"""
Telegram-бот для поддержки студентов в группах с возможностью обработки текста и изображений.
Использует GPT-4 Vision API для анализа изображений и ответов на вопросы.

Основные возможности:
- Обработка текстовых сообщений с обращением к боту
- Анализ изображений через Vision API
- Поддержка контекста сообщений
- Ограничение количества сообщений в сутки
- Асинхронная обработка запросов
"""

import json
import base64
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from telegram import Update
from telegram.ext import Application, MessageHandler, filters
from openai import AsyncOpenAI
from telegram.constants import ParseMode


from settings import (
    ALLOWED_CHATS,
    MAX_MESSAGES_PER_DAY,
    CONTEXT_EXPIRATION_MINUTES,
    CONTEXT_MESSAGE_LIMIT,
    JSON_LOG_FILE,
    OPEN_AI_API_KEY,
    BOT_TOKEN,
    BOT_ROLE,
    MODEL,
    BOT_USERNAME,
)

# Настройка логгера
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Глобальные переменные
chat_logs: Dict[int, Dict[int, List[Dict[str, str]]]] = {}
message_counters: Dict[int, Dict[int, int]] = {}

# Инициализация клиента OpenAI
openai_client = AsyncOpenAI(api_key=OPEN_AI_API_KEY)

# Создание временной директории для изображений
TEMP_IMAGE_DIR = Path("temp_images")
TEMP_IMAGE_DIR.mkdir(exist_ok=True)

# Для временного хранения изображений media_groups
media_groups: Dict[str, List[Update]] = {}  # Добавьте эту строку


async def encode_image(image_path: str) -> str:
    """Кодирует изображение в base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def save_temp_image(photo_file: Any, file_name: str) -> Path:
    """Сохраняет изображение во временную директорию."""
    file_path = TEMP_IMAGE_DIR / file_name
    await photo_file.download_to_drive(file_path)
    return file_path


async def cleanup_temp_images(image_paths: List[Path]) -> None:
    """Удаляет временные файлы изображений."""
    for path in image_paths:
        try:
            path.unlink()
        except Exception as e:
            logger.error(f"Ошибка при удалении {path}: {e}")


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
    """Возвращает контекст последних сообщений пользователя."""
    if chat_id in chat_logs and user_id in chat_logs[chat_id]:
        context = []
        for msg in chat_logs[chat_id][user_id][-CONTEXT_MESSAGE_LIMIT:]:
            if time.time() - msg["timestamp"] < CONTEXT_EXPIRATION_MINUTES * 60:
                context.append(msg)
        return context
    return []


async def update_user_context(
    chat_id: int, user_id: int, message: str, role: str = "user"
) -> None:
    """Обновляет контекст пользователя новым сообщением."""
    timestamp = time.time()
    if chat_id not in chat_logs:
        chat_logs[chat_id] = {}
    if user_id not in chat_logs[chat_id]:
        chat_logs[chat_id][user_id] = []
    chat_logs[chat_id][user_id].append(
        {
            "timestamp": timestamp,
            "message": message,
            "human_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "role": role,
        }
    )
    await save_chat_logs()


async def is_allowed_chat(chat_id: int) -> bool:
    """Проверяет, разрешен ли чат для работы бота."""
    return chat_id in ALLOWED_CHATS


async def is_admin(user_status: str) -> bool:
    """Проверяет, является ли пользователь администратором."""
    return user_status in ["creator", "administrator"]


async def check_message_limit(chat_id: int, user_id: int, is_admin: bool) -> bool:
    """Проверяет, не превышен ли лимит сообщений для пользователя."""
    if is_admin:
        return True
    return message_counters.get(chat_id, {}).get(user_id, 0) <= MAX_MESSAGES_PER_DAY


async def prepare_content_for_api(
    message_text: str, photos: Optional[List[Any]] = None
) -> Tuple[List[Dict[str, Any]], List[Path]]:
    content = [{"type": "text", "text": message_text}]
    image_paths = []

    if photos:
        logger.info(f"Обработка {len(photos)} изображений")
        for i, photo in enumerate(photos):
            try:
                file = await photo.get_file()
                file_path = await save_temp_image(file, f"temp_image_{i}.jpg")
                image_paths.append(file_path)
                logger.info(f"Сохранено изображение {i+1}: {file_path}")

                base64_image = await encode_image(str(file_path))
                content.append(
                    {
                        "type": "image",
                        "image_data": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "auto",
                        },
                    }
                )
                logger.info(f"Изображение {i+1} закодировано в base64")
            except Exception as e:
                logger.error(f"Ошибка при обработке изображения {i+1}: {e}")

    return content, image_paths


async def get_ai_response(
    context_messages: List[Dict[str, str]], content: List[Dict[str, Any]]
) -> str:
    """Получает ответ от OpenAI API."""
    messages = [{"role": "system", "content": BOT_ROLE}]

    # Добавляем контекст предыдущих сообщений
    for msg in context_messages:
        messages.append({"role": msg.get("role", "user"), "content": msg["message"]})

    # Добавляем текущее сообщение с контентом
    messages.append({"role": "user", "content": content})

    response = await openai_client.chat.completions.create(
        messages=messages, model=MODEL
    )
    return response.choices[0].message.content


async def send_response(update: Update, reply_text: str) -> None:
    """Отправляет ответ пользователю, разбивая длинные сообщения на части."""
    max_length = 4096
    reply_chunks = [
        reply_text[i : i + max_length] for i in range(0, len(reply_text), max_length)
    ]
    for chunk in reply_chunks:
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)


async def handle_message(update: Update, context) -> None:
    try:
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id

        logger.info(
            f"""
        ====== Новое сообщение ======
        Chat ID: {chat_id}
        User ID: {user_id}
        Has Photo: {bool(update.message.photo)}
        Caption: {update.message.caption}
        Text: {update.message.text}
        Media Group ID: {update.message.media_group_id}
        ===========================
        """
        )

        if not await is_allowed_chat(chat_id):
            logger.info(f"Чат {chat_id} не в списке разрешенных")
            return

        if update.effective_chat.type == "private":
            return

        user_status = (await update.effective_chat.get_member(user_id)).status
        user_is_admin = await is_admin(user_status)

        media_group_id = update.message.media_group_id

        if media_group_id:
            # Если сообщение является частью медиа-группы
            if media_group_id not in media_groups:
                media_groups[media_group_id] = []

            media_groups[media_group_id].append(update)

            # Ждем небольшое время, чтобы собрать все сообщения из группы
            await asyncio.sleep(1)

            # Проверяем, что мы собрали все сообщения
            updates = media_groups.pop(media_group_id, [])

            # Используем подпись первого сообщения как текст обращения
            first_update = updates[0]
            if (
                not first_update.message.caption
                or not first_update.message.caption.startswith(
                    f"@{context.bot.username}"
                )
            ):
                logger.info("Медиа-группа не содержит обращения к боту")
                return

            message_text = first_update.message.caption.replace(
                f"@{context.bot.username}", ""
            ).strip()
            photos = []
            for update in updates:
                if update.message.photo:
                    photos.extend(
                        update.message.photo[-1:]
                    )  # Добавляем фотографии с наивысшим разрешением

        else:
            # Если это не медиа-группа, обрабатываем как обычно
            # Определяем текст сообщения (из caption если есть фото, иначе из text)
            message_text = (
                update.message.caption
                if update.message.photo
                else update.message.text or ""
            )
            photos = update.message.photo

            if not message_text.startswith(f"@{context.bot.username}"):
                logger.info("Сообщение не начинается с обращения к боту")
                return

            message_text = message_text.replace(f"@{context.bot.username}", "").strip()

        if not await check_message_limit(chat_id, user_id, user_is_admin):
            await update.message.reply_text("Вы превысили лимит сообщений на сегодня.")
            return

        # Подготовка контента для API
        logger.info("Начало подготовки контента для API")
        content, image_paths = await prepare_content_for_api(message_text, photos)
        logger.info(f"Контент подготовлен. Изображений: {len(image_paths)}")

        # Получение контекста и ответа
        context_messages = await get_user_context(chat_id, user_id)
        logger.info("Получен контекст пользователя")

        reply_text = await get_ai_response(context_messages, content)
        logger.info("Получен ответ от AI")

        # Отправка ответа и обновление контекста
        await send_response(update, reply_text)
        await update_user_context(chat_id, user_id, reply_text, "assistant")
        logger.info("Ответ отправлен и контекст обновлен")

        # Очистка временных файлов
        if image_paths:
            await cleanup_temp_images(image_paths)
            logger.info("Временные файлы изображений очищены")

    except Exception as e:
        logger.exception(f"Ошибка при обработке сообщения: {e}")


async def reset_message_counters(context) -> None:
    """Сбрасывает счетчики сообщений."""
    global message_counters
    message_counters = {}
    logger.info("Счетчики сообщений сброшены.")


async def process_user_request(
    update: Update, context, message_text: str, photos: List[Any]
) -> None:
    try:
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id

        user_status = (await update.effective_chat.get_member(user_id)).status
        user_is_admin = await is_admin(user_status)

        if not await check_message_limit(chat_id, user_id, user_is_admin):
            await update.message.reply_text("Вы превысили лимит сообщений на сегодня.")
            return

        # Подготовка контента для API
        logger.info("Начало подготовки контента для API")
        content, image_paths = await prepare_content_for_api(message_text, photos)
        logger.info(f"Контент подготовлен. Изображений: {len(image_paths)}")

        # Получение контекста и ответа
        context_messages = await get_user_context(chat_id, user_id)
        logger.info("Получен контекст пользователя")

        reply_text = await get_ai_response(context_messages, content)
        logger.info("Получен ответ от AI")

        # Отправка ответа и обновление контекста
        await send_response(update, reply_text)
        await update_user_context(chat_id, user_id, reply_text, "assistant")
        logger.info("Ответ отправлен и контекст обновлен")

        # Очистка временных файлов
        if image_paths:
            await cleanup_temp_images(image_paths)
            logger.info("Временные файлы изображений очищены")

    except Exception as e:
        logger.exception(f"Ошибка при обработке запроса пользователя: {e}")


async def process_media_group(updates: List[Update], context) -> None:
    try:
        first_update = updates[0]
        chat_id = first_update.effective_chat.id
        user_id = first_update.effective_user.id

        user_status = (await first_update.effective_chat.get_member(user_id)).status
        user_is_admin = await is_admin(user_status)

        if not await check_message_limit(chat_id, user_id, user_is_admin):
            await first_update.message.reply_text(
                "Вы превысили лимит сообщений на сегодня."
            )
            return

        # Используем подпись первого сообщения как текст обращения
        message_text = first_update.message.caption.replace(
            f"@{context.bot.username}", ""
        ).strip()

        # Собираем все фотографии из медиагруппы
        photos = []
        for update in updates:
            if update.message.photo:
                photos.append(update.message.photo[-1])  # Наивысшее разрешение

        # Передаем данные для обработки
        await process_user_request(first_update, context, message_text, photos)
        logger.info("Медиагруппа обработана успешно")
    except Exception as e:
        logger.exception(f"Ошибка при обработке медиагруппы: {e}")


async def handle_media_group(update: Update, context) -> None:
    try:
        media_group_id = update.message.media_group_id
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id

        # Инициализация списка сообщений группы
        if "media_groups" not in context.chat_data:
            context.chat_data["media_groups"] = {}

        media_groups = context.chat_data["media_groups"]

        if media_group_id not in media_groups:
            media_groups[media_group_id] = []

        media_groups[media_group_id].append(update)

        # Проверяем, собраны ли все сообщения группы
        total_messages_in_group = update.message.media_group_size
        if len(media_groups[media_group_id]) == total_messages_in_group:
            # Все сообщения группы собраны, начинаем обработку
            updates = media_groups.pop(media_group_id)

            # Используем подпись первого сообщения как текст обращения
            first_update = updates[0]
            message_text = first_update.message.caption.replace(
                f"@{context.bot.username}", ""
            ).strip()

            # Собираем все фотографии из группы
            photos = []
            for msg in updates:
                if msg.message.photo:
                    photos.append(
                        msg.message.photo[-1]
                    )  # Берем фото с наивысшим разрешением

            # Передаем данные для обработки
            await process_user_request(first_update, context, message_text, photos)
    except Exception as e:
        logger.exception(f"Ошибка при обработке медиагруппы: {e}")


def main():
    try:
        global chat_logs

        try:
            with open(JSON_LOG_FILE, "r", encoding="utf-8") as file:
                chat_logs = json.load(file)
        except FileNotFoundError:
            logger.info("JSON файл с логами не найден. Создаем новый.")
            chat_logs = {}

        application = Application.builder().token(BOT_TOKEN).build()

        # Добавляем два отдельных обработчика для большей надежности
        application.add_handler(
            MessageHandler(
                filters.PHOTO & filters.CaptionRegex(f"^@{BOT_USERNAME}"),
                handle_message,
            )
        )
        application.add_handler(
            MessageHandler(
                filters.TEXT & filters.Regex(f"^@{BOT_USERNAME}"),
                handle_message,
            )
        )

        logger.info(
            f"""
        Бот настроен и запущен:
        - Username: {BOT_USERNAME}
        - Разрешенные чаты: {ALLOWED_CHATS}
        - Лимит сообщений: {MAX_MESSAGES_PER_DAY}
        - Обработчики: TEXT и PHOTO с префиксом @{BOT_USERNAME}
        """
        )

        application.run_polling(allowed_updates=["message"])

    except Exception as e:
        logger.exception(f"Ошибка при запуске бота: {e}")


if __name__ == "__main__":
    main()
