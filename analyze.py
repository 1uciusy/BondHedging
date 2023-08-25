import bisect
import datetime
from collections import deque
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategy2 import Strategy

__all__ = ["analysis", "plot", "price_plot", "future_plot", "pnl_plot"]
plt.rcParams["font.sans-serif"] = ["Heiti TC"]
plt.interactive(False)


def indicator(series: Optional[np.ndarray], num_year):
    """
    :param series: 逐日累计收益率序列
    :param num_year: 序列区间总年数
    :return: 返回区间收益率，区间年化收益率，30、60、90日最大回撤，区间最大回撤
    """
    stack = {30: deque(), 60: deque(), 90: deque(), len(series) + 1: deque()}
    prev_max = {30: -float("inf"), 60: -float("inf"), 90: -float("inf"), len(series) + 1: -float("inf")}
    draw_down = {30: 0, 60: 0, 90: 0, len(series) + 1: 0}
    # 按时间顺序从前到后扫描累计收益率序列
    for i, v in enumerate(series):
        for n_day in stack.keys():  # 维护一个窗口长度为n_day的历史的最大收益率单调队列
            while stack[n_day] and stack[n_day][0][0] <= i - n_day:  # 队头距离当前日期是超过n_day，则抛弃
                stack[n_day].popleft()
            draw_down[n_day] = max(draw_down[n_day], prev_max[n_day] - v)  # 队头是n_day历史内最高收益率
            while stack[n_day] and stack[n_day][-1][1] < v:  # 从队尾向前扫描，抛弃所有小于当前收益率的历史数据
                stack[n_day].pop()
            stack[n_day].append((i, v))
            prev_max[n_day] = stack[n_day][0][1]
    return series[-1], series[-1] / num_year, draw_down[len(series) + 1], \
           draw_down[30], draw_down[60], draw_down[60],


def analysis(s: Strategy, period: List[datetime.datetime], show_baseline=True):
    num_year = len(s.profile["累计息差"]) / 365.25
    print("|收益类型|区间收益|区间年化收益|区间最大回撤|30日最大回撤|60日最大回撤|90日最大回撤|")
    if show_baseline:
        print("|息差收益|{0}|{1}|{2}|{3}|{4}|{5}|".format(
            *tuple(map(lambda x: f"{'%3.2f' % x}%", indicator(s.profile["累计息差"] / s.borrow_cash * 100, num_year)))
        ))
        print("|无套期收益|{0}|{1}|{2}|{3}|{4}|{5}|".format(
            *tuple(map(lambda x: f"{'%3.2f' % x}%",
                       indicator(s.profile["累计无套期收益"] / s.borrow_cash * 100, num_year)))
        ))
    print("|套期收益|{0}|{1}|{2}|{3}|{4}|{5}|".format(
        *tuple(map(lambda x: f"{'%3.2f' % x}%", indicator(s.profile["累计套期收益"] / s.borrow_cash * 100, num_year)))
    ))

    for year in range(period[0].year, period[-1].year + 1):
        start_idx = bisect.bisect_left(period, datetime.datetime.strptime(f"{year}-01-01", "%Y-%m-%d"))
        end_idx = bisect.bisect_left(period, datetime.datetime.strptime(f"{year + 1}-01-01", "%Y-%m-%d"))
        if end_idx >= len(period):
            end_idx = len(period) - 1
        start_dt = period[start_idx]
        end_dt = period[end_idx]
        year_profit = (s.profile['累计套期收益'][end_idx] - s.profile['累计套期收益'][start_idx])
        year_profit = year_profit / s.borrow_cash * 100 / (end_dt - start_dt).days * 365
        prev_max = -float("inf")
        year_draw_down = 0
        for v in s.profile["累计套期收益"][start_idx:end_idx]:
            year_draw_down = max(prev_max - v, year_draw_down)
            prev_max = max(prev_max, v)
        print(
            f"{year}年套期收益率/最大回撤: {'%3.2f' % year_profit}%/{'%3.2f' % (year_draw_down / s.borrow_cash * 100)}%")
    if not show_baseline:
        return indicator(s.profile["累计套期收益"] / s.borrow_cash * 100, num_year)
    for year in range(period[0].year, period[-1].year + 1):
        start_idx = bisect.bisect_left(period, datetime.datetime.strptime(f"{year}-01-01", "%Y-%m-%d"))
        end_idx = bisect.bisect_left(period, datetime.datetime.strptime(f"{year + 1}-01-01", "%Y-%m-%d"))
        if end_idx >= len(period):
            end_idx = len(period) - 1
        start_dt = period[start_idx]
        end_dt = period[end_idx]
        year_profit = s.profile['累计无套期收益'][end_idx] - s.profile['累计无套期收益'][start_idx]
        year_profit = year_profit / s.borrow_cash * 100 / (end_dt - start_dt).days * 365
        prev_max = -float("inf")
        year_draw_down = 0
        for v in s.profile["累计无套期收益"][start_idx:end_idx]:
            year_draw_down = max(prev_max - v, year_draw_down)
            prev_max = max(prev_max, v)
        print(
            f"{year}年无套期收益率/最大回撤: {'%3.2f' % year_profit}%/{'%3.2f' % (year_draw_down / s.borrow_cash * 100)}%")
    return indicator(s.profile["累计套期收益"] / s.borrow_cash * 100, num_year)


def plot(s: Strategy, period: List[datetime.datetime], fut: pd.DataFrame, nv: pd.DataFrame,
         plot_contents=("price", "profit", "fut_position")):
    cost = s.profile
    fut_position = s.fut_position_his
    fig, ax = plt.subplots(nrows=1, ncols=len(plot_contents))
    fig.set_size_inches(15, 6)
    fig.subplots_adjust(left=0.06, right=0.95)
    for i, v in enumerate(plot_contents):
        if v == "profit":
            ax[i].plot(period, cost["累计息差"] / s.borrow_cash * 100, label="累计息差占借用资金比率（%）")
            ax[i].plot(period, cost["累计无套期收益"] / s.borrow_cash * 100, label="累计无套期收益占借用资金比率（%）")
            ax[i].plot(period, cost["累计套期收益"] / s.borrow_cash * 100, label="累计套期收益占借用资金比率（%）")
            ax[i].set_ylabel("收益率（%）")
        elif v == "fut_position":
            ax[i].plot(period, fut_position[10], label="10年国债期货手数")
            ax[i].plot(period, fut_position[5], label="5年国债期货手数")
            ax[i].plot(period, fut_position[2], label="2年国债期货手数")
            ax[i].set_ylabel("手数（1手=票面100w元现货）")
        elif v == "price":
            ax[i].plot(fut["date"], fut["10年期货收盘价"], label="10年期货收盘价")
            ax[i].plot(fut["date"], fut["5年期货收盘价"], label="5年期货收盘价")
            ax[i].plot(fut["date"], fut["2年期货收盘价"], label="2年期货收盘价")
            for k, _ in s.position.bond:
                ax[i].plot(nv["date"], nv[k], label=k)
            ax[i].set_ylabel("价格（元）")
        ax[i].legend()
        ax[i].xaxis.set_major_locator(plt.IndexLocator(365, 0))
    plt.show()


def price_plot(bond, bond_info: pd.DataFrame, nv: pd.DataFrame, fut: pd.DataFrame):
    """期现价格走势绘图"""
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(left=0.1, right=0.9)
    ax.plot(nv["date"], nv[bond], label=bond)
    f = (2, 5, 10)
    if bond_info.loc[bond, "到期日"].year - bond_info.loc[bond, "起息日"].year <= 5:
        f = f[:2]
    for i in f:
        ax.plot(fut["date"], fut[f"{i}年期货收盘价"], label=f"{i}年国债收盘价")
    ax.set_ylabel("价格（元）")
    ax.legend()
    ax.xaxis.set_major_locator(plt.IndexLocator(365, 0))
    plt.show()


def future_plot(s: Strategy, fut: pd.DataFrame, f=10, ):
    """单合约价格、ytm、irr、资金利率绘图"""
    baseline = pd.read_excel("中债国债收益率曲线.xlsx", sheet_name="Sheet1")
    baseline = baseline.join(fut["date"], how="inner")

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.subplots_adjust(left=0.06, right=0.95)
    fig.set_size_inches(15, 6)
    ax[0].plot(fut["date"], baseline[f"中债国债到期收益率:{f}年"], label=f"{f}年国债收益率(%)")
    ax[0].plot(fut["date"], s.apply(lambda x: x[f"{f}年ytm"], ("future",)), label=f"{f}年国债期货YTM(%)")
    ax[0].xaxis.set_major_locator(plt.IndexLocator(365, 0))
    ax[0].legend()

    ax[1].plot(fut["date"], s.apply(lambda x: x[f"{f}年irr"], ("future",)), label=f"{f}年国债期货IRR(%)")
    ax[1].plot(fut["date"], s.apply(lambda x: x["银行间隔夜利率（%）"], ("npv",)), label=f"R001(%)")
    if f == 10:
        cover_date = list()
        cover_irr = list()
        for i, v in enumerate(fut["date"]):
            if s.apply(lambda x: x[f"{f}年irr"], ("future",))[i] >= \
                    s.apply(lambda x: x["银行间隔夜利率（%）"], ("npv",))[i]:
                cover_date.append(v)
                cover_irr.append(s.apply(lambda x: x[f"{f}年irr"], ("future",))[i])
        ax[1].scatter(cover_date, cover_irr, c="r")
    ax[1].xaxis.set_major_locator(plt.IndexLocator(365, 0))
    ax[1].legend()
    plt.show()


def pnl_plot(period, cash, **kwargs):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(left=0.1, right=0.9)
    for k, v in kwargs.items():
        ax.plot(period, v / cash * 100, label=k)
    ax.set_ylabel("收益率（%）")
    ax.legend()
    ax.xaxis.set_major_locator(plt.IndexLocator(365, 0))
    plt.show()
