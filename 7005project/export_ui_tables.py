"""Run this in the latest notebook using %run export_ui_tables.py -i is WRONG.

Correct usage: %run -i export_ui_tables.py
The -i flag exposes current notebook variables. This file does not fit models.
Prepared with ChatGPT assistance for the team to review and attribute.
"""
from pathlib import Path
import numpy as np
import pandas as pd

target = Path.cwd()
if not (target / "新新出入境数据（英文版）.csv").exists():
    if (target / "7005project" / "新新出入境数据（英文版）.csv").exists():
        target = target / "7005project"
    else:
        raise FileNotFoundError("请先切换到包含客流CSV的7005project目录。")

if "df_feat" in globals() and {"Cluster", "HK_Ratio", "Mainland_Ratio", "Weekend_Weekday_Ratio", "CV_Volatility"}.issubset(df_feat.columns):
    profiles = df_feat.copy()
    if "Control Point" not in profiles.columns:
        profiles = profiles.rename_axis("Control Point").reset_index()
    profiles.to_csv(target / "port_profiles.csv", index=False, encoding="utf-8-sig")
    print("已导出 port_profiles.csv", len(profiles), "行")
else:
    print("未找到有效df_feat：请先运行最新Notebook的聚类单元。未生成画像文件。")

if "results" in globals() and isinstance(results, dict):
    tables = []
    for port, item in results.items():
        if not isinstance(item, dict) or not {"test_actual", "test_pred"}.issubset(item):
            continue
        actual = item["test_actual"]
        predicted = np.asarray(item["test_pred"]).reshape(-1)
        if len(actual) != len(predicted) or not isinstance(actual.index, pd.DatetimeIndex):
            raise ValueError(f"{port} 的预测长度或日期索引不匹配。")
        tables.append(pd.DataFrame({"Date":actual.index.strftime("%Y-%m-%d"), "Control_Point":port, "Direction":"Arrival", "Passenger_Type":"Total", "Model":"OLS", "Actual":np.asarray(actual), "Predicted":predicted}))
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(target / "regression_predictions.csv", index=False, encoding="utf-8-sig")
        print("已导出 regression_predictions.csv；来源为当前Notebook的OLS入境合计测试结果。")
    else:
        print("results中没有匹配的回归结果。未生成回归文件。")
else:
    print("未找到results：请先运行Notebook中存储results[cp]的完整OLS循环。")
