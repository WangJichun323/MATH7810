"""CrossFlow HK presentation layer.

Integration scaffold prepared with ChatGPT assistance for Ziyi to review and adapt.
Existing team model implementations remain in the submitted notebook. This app
reads their outputs; it does not claim to train or replace those algorithms.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="CrossFlow HK", layout="wide")
st.title("CrossFlow HK")
st.caption("香港口岸客流分析 · Passenger traffic analysis")
page = st.sidebar.radio("功能 Navigation", ["历史客流 History", "高峰查询 Peak forecast", "口岸画像 Profiles", "模型回看 Evaluation"])

def load_table(name, required, date_format=None):
    """Read a team output with explicit validation, without modifying its bytes."""
    path = ROOT / name
    if not path.exists():
        st.info(f"尚未提供 {name}。请将结果 CSV 放到 app.py 所在文件夹。")
        st.stop()
    try:
        table = pd.read_csv(path)
        absent = sorted(set(required) - set(table.columns))
        if absent:
            raise ValueError(f"缺少字段 {absent}")
        if table.empty:
            raise ValueError("文件没有记录")
        if date_format:
            table["Date"] = pd.to_datetime(table["Date"], format=date_format, errors="raise")
        return table
    except Exception as error:
        st.error(f"{name} 无法读取：{error}")
        st.stop()

def download(table, name):
    """Export exactly the selection visible to the user."""
    st.download_button("下载当前结果 Download CSV", table.to_csv(index=False).encode("utf-8-sig"), name, "text/csv")

def numeric(table, columns):
    """Reject missing and nonfinite numerical outputs rather than fabricating data."""
    for column in columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
        if not np.isfinite(table[column]).all():
            st.error(f"{column} 包含缺失或无效数字，请先检查源表。")
            st.stop()
    return table

if page == "历史客流 History":
    fields = ["Hong Kong Residents", "Mainland Visitors", "Other Visitors", "Total"]
    data = load_table("新新出入境数据（英文版）.csv", ["Date", "Control Point", "Arrival / Departure"] + fields, "%d-%m-%Y")
    data = numeric(data, fields)
    data = data[data.Date.between("2023-11-13", "2026-10-12")].copy()
    if data.empty:
        st.error("研究窗口内没有记录。")
        st.stop()
    if data.duplicated(["Date", "Control Point", "Arrival / Departure"]).any():
        st.error("日期×口岸×方向存在重复记录，请先检查。")
        st.stop()
    if (data[fields] < 0).any().any() or not np.allclose(data[fields[:3]].sum(axis=1), data.Total):
        st.error("人数为负数，或三类人次之和不等于 Total。")
        st.stop()
    st.caption("历史窗口 2023-11-13—2026-10-12；数据为人次，不是去重人数。")
    port = st.selectbox("口岸 Control Point", sorted(data["Control Point"].unique()))
    direction = st.selectbox("方向 Direction", ["Arrival", "Departure"])
    group = st.selectbox("客群 Passenger type", fields, index=3)
    dates = st.date_input("日期范围 Date range", (data.Date.min().date(), data.Date.max().date()), min_value=data.Date.min().date(), max_value=data.Date.max().date())
    if len(dates) != 2:
        st.info("请选完整的起止日期。")
        st.stop()
    start, end = map(pd.Timestamp, dates)
    selected = data[(data["Control Point"] == port) & (data["Arrival / Departure"] == direction) & data.Date.between(start, end)].sort_values("Date")
    if selected.empty:
        st.info("该选择没有记录。")
        st.stop()
    expected = (end-start).days + 1
    if len(selected) != expected:
        st.warning(f"日历天数 {expected}，有效记录 {len(selected)}；缺记录未补零。")
    a,b,c,d = st.columns(4)
    a.metric("有效天数", len(selected))
    b.metric("日均人次", f"{selected[group].mean():,.2f}")
    c.metric("中位数", f"{selected[group].median():,.2f}")
    d.metric("最大值", f"{selected[group].max():,.0f}")
    series = selected.set_index("Date")[group].reindex(pd.date_range(start, end))
    series.index.name = "Date"
    st.line_chart(series.rename(group))
    st.caption("无记录日期保留为空，不自动补成零。")
    stats = pd.DataFrame({"Metric":["Minimum", "Population variance (ddof=0)", "Sample variance (ddof=1)"], "Value":[selected[group].min(), selected[group].var(ddof=0), selected[group].var(ddof=1)]})
    st.dataframe(stats, hide_index=True)
    weekdays = selected.assign(Weekday=selected.Date.dt.dayofweek).groupby("Weekday")[group].mean().reindex(range(7))
    weekdays.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    st.subheader("星期效应 Weekday pattern")
    st.bar_chart(weekdays)
    with st.expander("查看记录 Records"):
        st.dataframe(selected, hide_index=True)
    download(selected, "historical_selection.csv")

elif page == "高峰查询 Peak forecast":
    cols = ["LR_Probability", "MLP_Probability"]
    data = numeric(load_table("Future_10Days_Prediction.csv", ["Date", "Control_Point"] + cols, "%Y-%m-%d"), cols)
    if not data[cols].apply(lambda x:x.between(0,1)).all().all() or data.duplicated(["Date", "Control_Point"]).any():
        st.error("概率超出0—1范围或同日同口岸有重复记录。")
        st.stop()
    st.info("本仓库当前版本以截至 2026-10-12 的历史数据生成预测；不是今天更新的实时预测。替换模型输出后，应同时核实并更新这条来源说明。")
    st.caption("入境＋出境合计；概率针对各口岸自身历史高峰，不等于排队时间、拥堵程度或安全容量。")
    st.write(f"文件覆盖：{data.Date.min():%Y-%m-%d}—{data.Date.max():%Y-%m-%d}")
    if data.Date.max().date() < pd.Timestamp.now(tz="Asia/Hong_Kong").date():
        st.warning("文件预测日期均已过去；当前展示为历史版本预测查询。")
    data = data[data.Control_Point != "Lok Ma Chau"]
    if data.empty:
        st.info("无可展示记录。")
        st.stop()
    day = st.selectbox("预测日期 Target date", sorted(data.Date.dt.date.unique()))
    daily = data[data.Date.dt.date == day]
    port = st.selectbox("口岸 Control Point", sorted(daily.Control_Point.unique()))
    row = daily[daily.Control_Point == port].iloc[0]
    a,b = st.columns(2)
    a.metric("Logistic peak probability", f"{row.LR_Probability:.1%}")
    b.metric("MLP peak probability", f"{row.MLP_Probability:.1%}")
    st.write("按0.5概率阈值分类：", "Logistic 高峰" if row.LR_Probability>=.5 else "Logistic 非高峰", "；", "MLP 高峰" if row.MLP_Probability>=.5 else "MLP 非高峰")
    st.caption("原落马洲公路口岸不作为未来出行选择；其余营业状态也需另行核实。模型不提供路线最优推荐。")
    st.subheader("所选口岸的预测日期序列")
    st.line_chart(data[data.Control_Point == port].set_index("Date")[cols])
    st.dataframe(daily, hide_index=True)
    download(daily, "peak_forecast_selection.csv")

elif page == "口岸画像 Profiles":
    fields = ["HK_Ratio", "Mainland_Ratio", "Weekend_Weekday_Ratio", "CV_Volatility"]
    data = numeric(load_table("port_profiles.csv", ["Control Point", "Cluster"] + fields), fields)
    st.caption("团队手写 K-Means++ 的历史样本聚类；Cluster编号本身没有大小顺序或官方业务含义。当前特征合并入境与出境。")
    port = st.selectbox("口岸 Control Point", sorted(data["Control Point"].unique()))
    row = data[data["Control Point"] == port].iloc[0]
    a,b,c = st.columns(3)
    a.metric("Cluster", str(int(row.Cluster)))
    b.metric("香港居民占比", f"{row.HK_Ratio:.1%}")
    c.metric("内地访客占比", f"{row.Mainland_Ratio:.1%}")
    st.write("周末／平日均值比：", round(row.Weekend_Weekday_Ratio,3))
    st.write("波动系数 CV：", round(row.CV_Volatility,3))
    st.subheader("同一聚类中的口岸")
    st.dataframe(data[data.Cluster == row.Cluster], hide_index=True)
    with st.expander("全部口岸画像"):
        st.dataframe(data, hide_index=True)
    download(data, "port_profiles_download.csv")

else:
    task = st.selectbox("查看任务", ["Classification", "Regression"])
    if task == "Classification":
        cols = ["Actual_Traffic", "Is_Overload_Actual", "LR_Probability", "MLP_Probability"]
        data = numeric(load_table("ControlPoint_Classification_Predictions.csv", ["Date", "Control_Point"] + cols, "%Y-%m-%d"), cols)
        st.caption("历史留出集结果，入境＋出境；当前源代码用全窗口P85生成标签，严格回测应改用训练期阈值后重新导出。")
        port = st.selectbox("口岸", sorted(data.Control_Point.unique()))
        selected = data[data.Control_Point == port].sort_values("Date")
        st.line_chart(selected.set_index("Date")[["LR_Probability", "MLP_Probability"]])
        metrics = []
        truth = selected.Is_Overload_Actual.to_numpy().astype(int)
        for model in ["LR", "MLP"]:
            pred = (selected[f"{model}_Probability"].to_numpy() >= .5).astype(int)
            tp = int(((pred==1)&(truth==1)).sum())
            fp = int(((pred==1)&(truth==0)).sum())
            fn = int(((pred==0)&(truth==1)).sum())
            metrics.append({"Model":model, "Accuracy":float((pred==truth).mean()), "Precision":tp/(tp+fp) if tp+fp else 0., "Recall":tp/(tp+fn) if tp+fn else 0., "F1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., "Actual peaks":int(truth.sum()), "Predicted peaks":int(pred.sum())})
        st.dataframe(pd.DataFrame(metrics), hide_index=True)
        st.caption("Precision/Recall分母为零时本页记为0；不能仅凭Accuracy评估少数高峰识别效果。")
        st.dataframe(selected, hide_index=True)
    else:
        data = numeric(load_table("regression_predictions.csv", ["Date", "Control_Point", "Direction", "Passenger_Type", "Model", "Actual", "Predicted"], "%Y-%m-%d"), ["Actual", "Predicted"])
        st.caption("历史回测结果，不是未来人数预测；按真实模型名称展示。")
        for col in ["Model", "Direction", "Passenger_Type", "Control_Point"]:
            value = st.selectbox(col, sorted(data[col].unique()))
            data = data[data[col] == value]
        data = data.sort_values("Date")
        error = data.Predicted-data.Actual
        a,b = st.columns(2)
        a.metric("MAE", f"{error.abs().mean():,.2f}")
        b.metric("RMSE", f"{np.sqrt((error**2).mean()):,.2f}")
        st.line_chart(data.set_index("Date")[["Actual", "Predicted"]])
        st.dataframe(data, hide_index=True)
    download(selected if task == "Classification" else data, "evaluation_selection.csv")
