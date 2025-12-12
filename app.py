import io
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DEFAULT_SAMPLE_PATH = Path(__file__).parent / "693a98fe3452e5558c60b808.csv"
BYTES_IN_MB = 1024 * 1024


st.set_page_config(page_title="Анализ трафика по IMEI", layout="wide")


@st.cache_data(show_spinner=False)
def load_table(file_content: bytes, file_name: str) -> pd.DataFrame:
    """Читает CSV или Excel по расширению."""
    suffix = Path(file_name).suffix.lower()
    buffer = io.BytesIO(file_content)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)


def expand_payload(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for payload in df["payload"]:
        try:
            obj = json.loads(payload)
        except Exception:
            continue

        ready_at = pd.to_datetime(obj.get("ready_at"), unit="ms", errors="coerce")
        values = obj.get("values", {}) or {}
        imei = values.get("imei")
        data = values.get("traffic_log_full_data") or []
        if not data or len(data) < 2:
            continue

        headers = data[0]
        for row in data[1:]:
            if len(row) != len(headers):
                continue
            rec = dict(zip(headers, row))
            rec.update(
                {
                    "ready_at": ready_at,
                    "imei": imei,
                    "traffic_from": pd.to_datetime(
                        values.get("traffic_log_full_from_ts"),
                        unit="ms",
                        errors="coerce",
                    ),
                    "traffic_to": pd.to_datetime(
                        values.get("traffic_log_full_to_ts"),
                        unit="ms",
                        errors="coerce",
                    ),
                }
            )
            records.append(rec)

    if not records:
        return pd.DataFrame()

    expanded = pd.DataFrame(records)
    for col in ["bytes_in", "bytes_out", "packets_in", "packets_out"]:
        if col in expanded.columns:
            expanded[col] = pd.to_numeric(expanded[col], errors="coerce")

    if "bytes_in" in expanded.columns:
        expanded["bytes_in_mb"] = expanded["bytes_in"] / BYTES_IN_MB
    if "bytes_out" in expanded.columns:
        expanded["bytes_out_mb"] = expanded["bytes_out"] / BYTES_IN_MB

    expanded["ready_at"] = pd.to_datetime(expanded["ready_at"], errors="coerce")
    return expanded.sort_values("ready_at")


@st.cache_data(show_spinner=False)
def load_expanded(file_content: bytes, file_name: str) -> pd.DataFrame:
    raw = load_table(file_content, file_name)
    return expand_payload(raw)


def main() -> None:
    st.title("Дешборд трафика по IMEI")
    st.caption(
        "Загрузите CSV или Excel с колонкой `payload` (JSON). "
        "Инструмент развернёт вложенные данные и построит графики. "
        "Все объёмы считаются в мегабайтах (MiB)."
    )

    with st.sidebar:
        st.header("Данные")
        uploaded = st.file_uploader("CSV/Excel файл", type=["csv", "xlsx", "xls"])
        use_sample = st.button("Использовать встроенный пример", type="secondary")

    data: pd.DataFrame | None = None
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        data = load_expanded(file_bytes, uploaded.name)
    elif use_sample and DEFAULT_SAMPLE_PATH.exists():
        data = load_expanded(DEFAULT_SAMPLE_PATH.read_bytes(), DEFAULT_SAMPLE_PATH.name)
        st.toast("Загружен встроенный пример")
    else:
        st.info(
            "Загрузите CSV/XLSX с трафиком или нажмите кнопку примера в сайдбаре. "
            "Формат: одна колонка `payload` с JSON, где есть imei и "
            "traffic_log_full_data."
        )
        return

    if data is None or data.empty:
        st.warning("Не удалось разобрать данные. Проверьте формат CSV/XLSX.")
        return

    imeis = sorted(data["imei"].dropna().unique())
    protocols = sorted(data["protocol"].dropna().unique())

    with st.sidebar:
        st.header("Фильтры")
        selected_imeis = st.multiselect(
            "IMEI", options=imeis, default=imeis if imeis else None
        )
        selected_protocols = st.multiselect(
            "Протокол", options=protocols, default=protocols if protocols else None
        )
        ips = sorted(data["ip"].dropna().unique())
        selected_ips = st.multiselect(
            "IP адреса", options=ips, default=ips if ips else None
        )
        metric_choice = st.radio(
            "Метрика для графиков",
            options=["all", "bytes_out", "bytes_in"],
            format_func=lambda x: {
                "all": "Все (in+out)",
                "bytes_out": "Только исходящий (out)",
                "bytes_in": "Только входящий (in)",
            }[x],
        )
        time_min, time_max = data["ready_at"].min(), data["ready_at"].max()
        if pd.notnull(time_min) and pd.notnull(time_max):
            start, end = st.slider(
                "Время готовности (ready_at)",
                min_value=time_min.to_pydatetime(),
                max_value=time_max.to_pydatetime(),
                value=(time_min.to_pydatetime(), time_max.to_pydatetime()),
            )
        else:
            start, end = None, None

    filtered = data.copy()
    if selected_imeis:
        filtered = filtered[filtered["imei"].isin(selected_imeis)]
    if selected_protocols:
        filtered = filtered[filtered["protocol"].isin(selected_protocols)]
    if selected_ips:
        filtered = filtered[filtered["ip"].isin(selected_ips)]
    if start and end:
        filtered = filtered[
            (filtered["ready_at"] >= pd.Timestamp(start))
            & (filtered["ready_at"] <= pd.Timestamp(end))
        ]

    if filtered.empty:
        st.warning("После фильтрации данных не осталось.")
        return

    # Сохраняем все данные для графика динамики (до фильтрации по максимальному трафику)
    filtered_all_reports = filtered.copy()

    # Отчеты накапливают данные за день, нужно брать отчет с максимальным трафиком для каждого IMEI
    # Сначала считаем суммарный трафик для каждого отчета (IMEI + ready_at)
    filtered["date"] = filtered["ready_at"].dt.date
    filtered["total_traffic_mb"] = (
        filtered.get("bytes_out_mb", pd.Series(dtype=float)).fillna(0)
        + filtered.get("bytes_in_mb", pd.Series(dtype=float)).fillna(0)
    )
    
    # Группируем по IMEI + ready_at (это один отчет) и суммируем трафик
    report_totals = (
        filtered.groupby(["imei", "ready_at", "date"])["total_traffic_mb"]
        .sum()
        .reset_index()
    )
    
    # Для каждого IMEI + дата находим ready_at отчета с максимальным трафиком
    max_reports_idx = (
        report_totals.groupby(["imei", "date"])["total_traffic_mb"]
        .idxmax()
    )
    selected_reports = report_totals.loc[max_reports_idx.values, ["imei", "ready_at", "date"]]
    
    # Оставляем только строки из выбранных отчетов (для суммирования)
    filtered = filtered.merge(
        selected_reports[["imei", "ready_at", "date"]],
        on=["imei", "ready_at", "date"],
        how="inner"
    )
    filtered = filtered.drop(columns=["date", "total_traffic_mb"], errors="ignore")

    if metric_choice == "bytes_out":
        metric_cols = ["bytes_out_mb"]
        metric_sort = "bytes_out_mb"
        metric_label = "Исходящий (МБ)"
    elif metric_choice == "bytes_in":
        metric_cols = ["bytes_in_mb"]
        metric_sort = "bytes_in_mb"
        metric_label = "Входящий (МБ)"
    else:
        metric_cols = ["bytes_out_mb", "bytes_in_mb"]
        metric_sort = "total_mb"
        metric_label = "Суммарно (МБ)"
        filtered["total_mb"] = filtered[["bytes_out_mb", "bytes_in_mb"]].sum(axis=1)

    total_out_mb = filtered.get("bytes_out_mb", pd.Series(dtype=float)).sum()
    total_in_mb = filtered.get("bytes_in_mb", pd.Series(dtype=float)).sum()
    total_packets = filtered.get("packets_out", pd.Series(dtype=float)).sum() + filtered.get(
        "packets_in", pd.Series(dtype=float)
    ).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("МБ отправлено", f"{total_out_mb:,.2f}")
    c2.metric("МБ получено", f"{total_in_mb:,.2f}")
    c3.metric("Пакетов всего", f"{total_packets:,.0f}")

    # Для графика динамики используем все отчеты (чтобы видеть всю динамику)
    # Для суммирования используем только отчеты с максимальным трафиком
    traffic_ts_data = filtered_all_reports.copy()
    if metric_choice == "all":
        traffic_ts_data["total_mb"] = traffic_ts_data[["bytes_out_mb", "bytes_in_mb"]].sum(axis=1)
        traffic_cols = ["bytes_out_mb", "bytes_in_mb"]
    else:
        traffic_cols = metric_cols
    
    traffic_ts = (
        traffic_ts_data.groupby(["ready_at", "imei"])[traffic_cols]
        .sum()
        .reset_index()
        .melt(
            id_vars=["ready_at", "imei"],
            var_name="metric",
            value_name="megabytes",
        )
    )
    fig_ts = px.line(
        traffic_ts,
        x="ready_at",
        y="megabytes",
        color="imei",
        line_dash="metric",
        markers=True,
        title="Динамика трафика (МБ, in/out)",
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    top_ip = filtered.groupby(["ip"])[metric_cols].sum().reset_index()
    if metric_choice == "all":
        top_ip["total_mb"] = top_ip[["bytes_out_mb", "bytes_in_mb"]].sum(axis=1)
    sort_col = metric_sort
    top_ip = top_ip.sort_values(sort_col, ascending=False).head(15)
    fig_ip = px.bar(
        top_ip,
        x="ip",
        y=metric_cols if len(metric_cols) > 1 else metric_cols[0],
        title=f"Топ IP по трафику ({metric_label})",
        barmode="group",
    )
    st.plotly_chart(fig_ip, use_container_width=True)

    port_stats = filtered.groupby(["port"])[metric_cols].sum().reset_index()
    if metric_choice == "all":
        port_stats["total_mb"] = port_stats[["bytes_out_mb", "bytes_in_mb"]].sum(axis=1)
    port_stats = port_stats.sort_values(sort_col, ascending=False)
    fig_port = px.bar(
        port_stats.head(15),
        x="port",
        y=metric_cols[0] if len(metric_cols) == 1 else sort_col,
        title=f"Распределение по портам ({metric_label})",
    )
    st.plotly_chart(fig_port, use_container_width=True)

    # Топ по IMEI за день (сортировка по трафику)
    st.subheader("Потребление по IMEI за день (сортировка по трафику)")
    daily = filtered.copy()
    daily["date"] = daily["ready_at"].dt.date
    daily_agg = daily.groupby(["date", "imei"])[metric_cols].sum().reset_index()
    if metric_choice == "all":
        daily_agg["total_mb"] = daily_agg[["bytes_out_mb", "bytes_in_mb"]].sum(axis=1)
    daily_agg = daily_agg.sort_values(["date", sort_col], ascending=[False, False])
    cols_to_show = ["date", "imei", sort_col]
    if len(metric_cols) > 1:
        for c in metric_cols:
            if c not in cols_to_show:
                cols_to_show.append(c)
    st.dataframe(
        daily_agg[cols_to_show],
        use_container_width=True,
        height=300,
    )

    # Heatmap IMEI ↔ IP
    st.subheader("Связка IMEI ↔ IP (heatmap)")
    ip_heat = filtered.groupby(["imei", "ip"])[metric_cols].sum().reset_index()
    if metric_choice == "all":
        ip_heat["value"] = ip_heat[["bytes_out_mb", "bytes_in_mb"]].sum(axis=1)
    else:
        ip_heat["value"] = ip_heat[metric_cols[0]]
    top_ips = (
        ip_heat.groupby("ip")["value"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
        .index
    )
    ip_heat = ip_heat[ip_heat["ip"].isin(top_ips)]
    heat_ip_pivot = (
        ip_heat.pivot_table(index="imei", columns="ip", values="value", fill_value=0)
        if not ip_heat.empty
        else pd.DataFrame()
    )
    if not heat_ip_pivot.empty:
        fig_heat_ip = px.imshow(
            heat_ip_pivot,
            labels=dict(x="IP", y="IMEI", color=metric_label),
            aspect="auto",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_heat_ip, use_container_width=True)
    else:
        st.info("Недостаточно данных для heatmap IMEI ↔ IP (после фильтров).")

    # Таблица пакетов по IMEI и IP
    st.subheader("Пакеты по IMEI и IP адресу")
    packets_by_imei_ip = (
        filtered.groupby(["imei", "ip"])
        .agg(
            {
                "packets_out": "sum",
                "packets_in": "sum",
                "bytes_out_mb": "sum",
                "bytes_in_mb": "sum",
            }
        )
        .reset_index()
    )
    packets_by_imei_ip["total_packets"] = (
        packets_by_imei_ip["packets_out"] + packets_by_imei_ip["packets_in"]
    )
    packets_by_imei_ip = packets_by_imei_ip.sort_values("total_packets", ascending=False)
    
    # Форматируем для отображения
    display_cols = ["imei", "ip", "packets_out", "packets_in", "total_packets"]
    if "bytes_out_mb" in packets_by_imei_ip.columns:
        display_cols.extend(["bytes_out_mb", "bytes_in_mb"])
    
    st.dataframe(
        packets_by_imei_ip[display_cols],
        use_container_width=True,
        height=400,
        column_config={
            "packets_out": st.column_config.NumberColumn("Пакетов отправлено", format="%d"),
            "packets_in": st.column_config.NumberColumn("Пакетов получено", format="%d"),
            "total_packets": st.column_config.NumberColumn("Всего пакетов", format="%d"),
            "bytes_out_mb": st.column_config.NumberColumn("МБ отправлено", format="%.2f"),
            "bytes_in_mb": st.column_config.NumberColumn("МБ получено", format="%.2f"),
        },
    )

    # Heatmap IMEI ↔ порт
    st.subheader("Связка IMEI ↔ порт (heatmap)")
    port_heat = filtered.groupby(["imei", "port"])[metric_cols].sum().reset_index()
    if metric_choice == "all":
        port_heat["value"] = port_heat[["bytes_out_mb", "bytes_in_mb"]].sum(axis=1)
    else:
        port_heat["value"] = port_heat[metric_cols[0]]
    top_ports = (
        port_heat.groupby("port")["value"]
        .sum()
        .sort_values(ascending=False)
        .head(30)
        .index
    )
    port_heat = port_heat[port_heat["port"].isin(top_ports)]
    heat_port_pivot = (
        port_heat.pivot_table(index="imei", columns="port", values="value", fill_value=0)
        if not port_heat.empty
        else pd.DataFrame()
    )
    if not heat_port_pivot.empty:
        fig_heat_port = px.imshow(
            heat_port_pivot,
            labels=dict(x="Порт", y="IMEI", color=metric_label),
            aspect="auto",
            color_continuous_scale="OrRd",
        )
        st.plotly_chart(fig_heat_port, use_container_width=True)
    else:
        st.info("Недостаточно данных для heatmap IMEI ↔ порт (после фильтров).")

    if "protocol" in filtered.columns:
        proto_stats = filtered.groupby("protocol")[metric_cols].sum().reset_index()
        if metric_choice == "all":
            proto_stats["total_mb"] = proto_stats[["bytes_out_mb", "bytes_in_mb"]].sum(
                axis=1
            )
        fig_proto = px.pie(
            proto_stats,
            names="protocol",
            values=metric_cols[0] if len(metric_cols) == 1 else sort_col,
            title=f"Доля протоколов ({metric_label})",
        )
        st.plotly_chart(fig_proto, use_container_width=True)

    # Дельта между последними двумя отчётами на IMEI
    st.subheader("Дельта между последними двумя отчётами (по IMEI)")
    report_totals = (
        filtered.groupby(["imei", "ready_at"])[metric_cols]
        .sum()
        .reset_index()
        .sort_values(["imei", "ready_at"])
    )
    if metric_choice == "all":
        report_totals["total_mb"] = report_totals[["bytes_out_mb", "bytes_in_mb"]].sum(axis=1)

    deltas = []
    for imei, grp in report_totals.groupby("imei"):
        if len(grp) < 2:
            continue
        last_two = grp.tail(2)
        curr, prev = last_two.iloc[-1], last_two.iloc[-2]
        delta_val = curr[sort_col] - prev[sort_col]
        delta_row = {
            "imei": imei,
            "last_ready_at": curr["ready_at"],
            "prev_ready_at": prev["ready_at"],
            "delta": delta_val,
        }
        if metric_choice == "all":
            delta_row["delta_out_mb"] = curr.get("bytes_out_mb", 0) - prev.get("bytes_out_mb", 0)
            delta_row["delta_in_mb"] = curr.get("bytes_in_mb", 0) - prev.get("bytes_in_mb", 0)
        deltas.append(delta_row)

    delta_df = pd.DataFrame(deltas)
    if not delta_df.empty:
        delta_df = delta_df.sort_values("delta", ascending=False)
        st.dataframe(
            delta_df,
            use_container_width=True,
            height=250,
        )
        fig_delta_bar = px.bar(
            delta_df,
            x="imei",
            y="delta",
            color="delta",
            color_continuous_scale="RdBu",
            title=f"Дельта последнего отчёта относительно предыдущего ({metric_label})",
        )
        st.plotly_chart(fig_delta_bar, use_container_width=True)
    else:
        st.info("Недостаточно данных (нужно минимум два отчёта на IMEI) для расчёта дельты.")

    # История дельт по отчётам (для выбранной метрики)
    st.subheader("История дельт по отчётам (последовательные отчёты на IMEI)")
    if not report_totals.empty and report_totals.groupby("imei").size().max() > 1:
        delta_col = sort_col
        report_totals["delta_prev"] = report_totals.groupby("imei")[delta_col].diff()
        delta_series = report_totals.dropna(subset=["delta_prev"])
        if not delta_series.empty:
            fig_delta_ts = px.line(
                delta_series,
                x="ready_at",
                y="delta_prev",
                color="imei",
                markers=True,
                title=f"Дельта между соседними отчётами ({metric_label})",
            )
            st.plotly_chart(fig_delta_ts, use_container_width=True)
        else:
            st.info("Нет последовательных отчётов для построения истории дельт.")
    else:
        st.info("Недостаточно отчётов для построения истории дельт.")

    st.subheader("Сырые данные после фильтров")
    st.dataframe(filtered, use_container_width=True, height=400)


if __name__ == "__main__":
    main()

