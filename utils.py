import datetime
from typing import List

import pandas as pd

__all__ = ["date_check", "data_prep", "payday_gen"]


def date_check(date) -> datetime.date:
    """
    :param date: 需要检查类型的日期
    :return: 类型检查后正确的日期格式
    """
    if type(date) == datetime.datetime:
        date = date.date()
    elif type(date) == pd.Timestamp:
        date = date.to_pydatetime().date()
    elif type(date) == datetime.date:
        pass
    elif type(date) == str:
        try:
            date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("date should be datetime.date, pd.Timestamp or string with format %Y-%m-%d")
    else:
        raise ValueError("date should be datetime.date, pd.Timestamp or string with format %Y-%m-%d")
    return date


def data_prep():
    """
    读取数据
    """
    dv = pd.read_excel("table.xlsx", sheet_name="dv")
    nv = pd.read_excel("table.xlsx", sheet_name="rate&nv")
    fut = pd.read_excel("table.xlsx", sheet_name="fut")
    code = pd.read_excel("table.xlsx", sheet_name="code")
    code_dict = dict()
    for row in code.iterrows():
        code_dict[row[1][0]] = row[1][1]
    bond = pd.read_excel("table.xlsx", sheet_name="int")
    ctd = pd.read_excel("table.xlsx", sheet_name="ctd")
    return dv, nv, fut, code, code_dict, bond, ctd


def payday_gen(start, end, freq: int) -> List[str]:
    """
    给定起息日、到期日、年付息次数，生成所有付息日
    :param start: 起息日
    :param end: 到期日
    :param freq: 年付息次数
    :return: 每个付息日
    """
    start = date_check(start).strftime("%Y-%m-%d")
    end = date_check(end).strftime("%Y-%m-%d")
    tmp = [start]
    if freq == 1:
        for i in range(int(start[:4]) + 1, int(end[:4]) + 1):
            tmp.append(start.replace(start[:4], str(i), 1))
    elif freq == 2:
        year = start[:4]
        month = start[5:7]
        for i in range(2 * (int(end[:4]) - int(start[:4]))):
            new_month = int(month) + (i + 1) * 6
            new_year = int(year) + new_month // 12
            new_month = new_month % 12
            new_year -= 1 if new_month == 0 else 0
            new_month = 12 if new_month == 0 else new_month
            new_year = str(new_year)
            new_month = str(new_month) if len(str(new_month)) == 2 else "0" + str(new_month)
            tmp_dt = f"{new_year}-{new_month}-{start[-2:]}"
            tmp.append(tmp_dt)
    else:
        raise ValueError(f"{freq} is not supported")
    return tmp


def process_gt(table: pd.DataFrame):
    """加工学习目标"""
    for f in (10, 5, 2):
        table[f"{f}_diff"] = table[f"{f}年期货收盘价"].diff(1).fillna(0)  # 后减前，小于0有收益，第一个值是null
        table[f"{f}_profit"] = table[f"{f}_diff"] < 0
    return table
