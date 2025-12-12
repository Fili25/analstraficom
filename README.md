# Анализ трафика по IMEI (Streamlit)

Локальный/облачный дашборд для CSV/XLSX с колонкой `payload` (JSON) из отчётов трафика. Поддерживает фильтры по IMEI, протоколу, IP, порту, переключение метрики (in/out/all), расчёт в мегабайтах, топы, heatmap IMEI↔IP/порт, дельты между отчётами.

## Локальный запуск
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.headless true --server.port 8501
```
Открыть в браузере `http://localhost:8501`.

## Структура
- `app.py` — логика загрузки, парсинга, фильтров и визуализации.
- `requirements.txt` — зависимости (pandas, streamlit, plotly, openpyxl).
- `.streamlit/config.toml` — базовый конфиг сервера (headless, 0.0.0.0, порт 8501).
- `693a98fe3452e5558c60b808.csv` — встроенный пример (можно удалить для публичного репо).

## Деплой в Streamlit Community Cloud
1. Создайте публичный (или приватный) репозиторий на GitHub. Добавьте файлы: `app.py`, `requirements.txt`, `.streamlit/config.toml`, опционально пример CSV.
2. Перейдите на https://streamlit.io/cloud → New app.
3. Выберите репозиторий, ветку и файл запуска `app.py`. Нажмите Deploy.
4. Дождитесь билда — получите публичный URL. Обновления: пуш в ту же ветку триггерит пересборку.

### Секреты / приватные данные
- Чувствительные данные не хранить в репозитории. Пользователи загружают свои CSV/XLSX через UI.
- Для ключей/API используйте Secrets в панели Streamlit Cloud (не `.streamlit/secrets.toml` в репо).

### Советы
- Для больших файлов используйте фильтры перед загрузкой или предагрегацию.
- Если выкладываете публично, уберите реальные данные из примера.

