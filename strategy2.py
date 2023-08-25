"""
重构后策略代码，批处理信号，流处理更新损益、更新模型
"""
import bisect
import datetime
from collections import defaultdict, deque
from typing import Optional, Dict, List, Iterable, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import date_check, data_prep, payday_gen

__all__ = ["Strategy"]


class Position(object):
    """现券、期货仓位管理"""

    def __init__(self, ):
        self.unit = 1e7
        self._bond = defaultdict(float)
        self._future = {10: 0, 5: 0, 2: 0}
        self._future_his = deque()

    def add_bond(self, bond_name, n_unit, ):
        """添加现券，1 unit = 1kw现券"""
        self._bond[bond_name] += self.unit * n_unit

    def update_future(self, new_position: Optional[dict] = None):
        """用新的仓位替换旧仓位"""
        self._future.update(new_position)

    def cache_future(self):
        """保存仓位"""
        self._future_his.append(self._future.copy())

    def get_future_history(self, start_date, end_date):
        """获取期货仓位历史信息"""
        res = pd.DataFrame(self._future_his).fillna(0)
        date = pd.date_range(start_date, end_date)
        res.index = date
        return res

    @property
    def bond(self):
        return self._bond.items()

    @property
    def future(self):
        return self._future.items()


class Market(object):
    """提供行情数据、信号计算"""

    def __init__(self, dt=datetime.datetime.strptime("2019-01-01", "%Y-%m-%d")):
        """初始化，会将第一个交易日的数据缓存"""
        dv, nv, fut, code, code_dict, bond_info, ctd_info = data_prep()
        bond_info = bond_info.set_index("简称")
        ctd_info = ctd_info.set_index("代码")
        self.bond_info = bond_info
        self.ctd_info = ctd_info
        self._table: Dict[str, pd.DataFrame] = {"dv": dv, "npv": nv, "future": fut}
        self._previous = defaultdict(dict)
        self._cur = defaultdict(dict)
        self._cur_date: Optional[datetime.datetime] = None
        self._trade_day = set(dv["date"])

        while dt not in self._trade_day:
            dt += datetime.timedelta(days=1)
        for k, v in self._table.items():
            self._cur[k] = v.loc[v["date"] == dt].iloc[0].to_dict()

        for f in (10, 5, 2):
            self._table["future"][f"{f}年ytm"] = 3 + (100 - self._table["future"][f"{f}年期货收盘价"]) / (f * 0.8)
            self._table["future"][f"{f}年irr"] = 0
        self._update_irr()

    def roll(self, dt: datetime.datetime):
        """逢交易日，将前一日数据轮动，读取当日数据，遇周末时，cur仍是周五数据"""
        self._cur_date = dt
        if self.is_trade_day():
            self._previous.update(self._cur)
            for k, v in self._table.items():
                self._cur[k] = v.loc[v["date"] == dt].iloc[0].to_dict()

    def apply(self, func: Callable, table_names: Iterable, replace_table="", inplace=False):
        """给操作数据表做批处理提供一个接口, 允许多表串联"""
        assert (replace_table not in self._table) ^ inplace  # 异或，如果inplace修改，替换的表就必须存在
        res = func(*(self._table[table_name] for table_name in table_names))
        if inplace:
            self._table[replace_table] = res
        return res

    def trend_break(self, setting: Dict[str, Iterable[str]], default=True, window=5, closed="left"):
        """计算突破信号"""
        for table_name, columns in setting.items():
            table = self._table[table_name]
            for c in columns:
                table[f"{c}_break_{window}"] = table[c] > table[c].rolling(window=window, closed=closed).max()
                table.loc[:window - 1, f"{c}_break_{window}"] = default  # loc里slice是右闭的
                # 中债估值更新最晚，在期货收盘后。损益计算方法要保证T+0调仓，需要提前预测今天中债估值 or 使用T-1突破[T-6,T-2]信号
                if table_name == "npv":
                    table[f"{c}_break_{window}"] = table[f"{c}_break_{window}"].shift(1).fillna(default)

    def is_trade_day(self):
        return self._cur_date in self._trade_day

    def _update_irr(self, ):
        """计算IRR"""
        # 获取交割日，设置为季月第二个周五
        start_year = date_check(self._table["future"]["date"].iloc[0]).year
        end_year = date_check(self._table["future"]["date"].iloc[-1]).year
        start_date = date_check(f"{start_year}-01-01")
        end_date = date_check(f"{end_year + 2}-01-01")
        deliver_dates: List[datetime.date] = []
        nth_week = 0
        for i in range((end_date - start_date).days + 1):
            dt = start_date + datetime.timedelta(days=i)
            if dt.month in (3, 6, 9, 12) and dt.weekday() == 5:
                nth_week += 1
                if nth_week == 2:
                    deliver_dates.append(dt)
                if deliver_dates and dt.month == deliver_dates[-1].month:
                    nth_week = 0
        # 计算所有CTD的付息日
        ctd_pay: Dict[str, List[str]] = dict()
        for code, (start, end, freq) in self.ctd_info[["起息日", "到期日", "年付息次数"]].iterrows():
            ctd_pay[code] = payday_gen(start, end, freq)
        # 计算每一天每个期货的IRR
        self._table["future"]["10_irr_cover"] = False
        self._table["future"]["5_irr_cover"] = False
        self._table["future"]["2_irr_cover"] = False
        for dt in self._table["future"]["date"]:
            # 取下一交割日，如果距离当日少于30天，认为合约已经换月，取下下一交割日
            idx = bisect.bisect_left(deliver_dates, date_check(dt))
            idx = idx if (deliver_dates[idx] - date_check(dt)).days >= 30 else idx + 1
            deliver_dt = deliver_dates[idx]

            idx = self._table["future"]["date"] == dt
            for f in (10, 5, 2):
                ctd_code = self._table["future"].loc[idx, f"{f}年期货CTD"].iloc[0]
                # ctd券上一付息日
                left = bisect.bisect_left(ctd_pay[ctd_code], date_check(dt).strftime("%Y-%m-%d"))
                previous_pay_day = ctd_pay[ctd_code][left - 1]
                # t时刻期货全价
                tmp = self._table["future"].loc[idx, f"{f}年期货收盘价"].iloc[0]
                tmp *= self._table["future"].loc[idx, f"{f}年转换因子"].iloc[0]  # 期货价格*cf
                ai = ((date_check(dt) - date_check(previous_pay_day)).days % 365) / 365  # 应计利息
                ai *= self.ctd_info["票面利率发行"].loc[ctd_code]
                tmp += ai
                # 现券全价-期货剩余期限内现券的付息
                tmp_ctd_pv = self._table["future"].loc[idx, f"{f}年CTD-估值全价"]  # 现货全价
                tmp_ctd_pv -= ((deliver_dt - date_check(dt)).days + 1) / 365 * self.ctd_info["票面利率发行"].loc[
                    ctd_code]
                irr = (tmp - tmp_ctd_pv) / tmp_ctd_pv * 365 / ((deliver_dt - date_check(dt)).days + 1) * 100
                self._table["future"].loc[idx, f"{f}年irr"] = irr
                self._table["future"].loc[idx, f"{f}_irr_cover"] = \
                    self._table["npv"].loc[idx, '银行间隔夜利率（%）'] < irr

    @property
    def today(self):
        return self._cur_date

    @property
    def pre_trade_day(self):
        return self._previous

    @property
    def cur(self):
        return self._cur

    @property
    def npv_cache(self):
        return self._table["npv"].loc[self._table["npv"]["date"] < self._cur_date]

    @property
    def future_cache(self):
        return self._table["future"].loc[self._table["future"]["date"] < self._cur_date]

    @property
    def dv_cache(self):
        return self._table["dv"].loc[self._table["dv"]["date"] < self._cur_date]


class Strategy(Market):
    def __init__(self, start_dt=datetime.datetime.strptime("2019-01-01", "%Y-%m-%d"),
                 inhibitor: Optional[Dict] = None, batch_size=10, verbose=False, ):
        super(Strategy, self).__init__(start_dt)
        # 条件抑制器，默认为false，启用时抑制某条规则
        self.inhibitor = defaultdict(bool)
        if inhibitor is not None:
            self.inhibitor.update(inhibitor)
        # 打印详细信息
        self.verbose = verbose
        # 历史收益信息
        self.profile = defaultdict(deque)
        # 期货仓位历史声明
        self.fut_position_his = pd.DataFrame()
        # 模型部分，四种异质的模型备选
        self.batch_size = batch_size
        self.model_zoo = {"NB": defaultdict(lambda: defaultdict(lambda: BernoulliNB(alpha=.1))),
                          "LR": defaultdict(lambda: defaultdict(lambda: LogisticRegression())),
                          "SVC": defaultdict(lambda: defaultdict(lambda: SVC())),
                          "Tree": defaultdict(lambda: defaultdict(lambda: DecisionTreeClassifier()))}
        self.model_hedge = {"NB": defaultdict(lambda: defaultdict(bool)),
                            "LR": defaultdict(lambda: defaultdict(bool)),
                            "SVC": defaultdict(lambda: defaultdict(bool)),
                            "Tree": defaultdict(lambda: defaultdict(bool)), }
        # 行情数据
        self.trend_break(setting={"future": ("10年期货收盘价", "5年期货收盘价", "2年期货收盘价",), }, )
        self.trend_break(setting={"npv": ('18北京债05', '15广东债18', "15北京债08", "18上海13"), }, )
        # 初始化仓位，配置一
        self.position = Position()
        self.position.add_bond("15广东债18", 5)
        self.position.add_bond("18北京债05", 5)
        # 配置二
        # self.position.add_bond("15北京债08", 5)
        # self.position.add_bond("18上海13", 5)
        self.borrow_cash = sum(scale for bond, scale in self.position.bond) * 1.1

    def handle_data(self, dt: datetime.datetime):
        """
        1. 现券计息、持仓各项资产计算资本利得
        2. 更新对冲头寸
        按照损益计算方式，用于实盘操作时，是T+0日临近收盘时间更新头寸
        调仓时，T+0中债估值还未更新，且地方债流动性差难以使用成交价作为估值，使用T-1日突破T-6到T-2的信号
        """
        self.roll(dt=dt)
        self.profit_rolling()  # 无论是否交易日均需运行，需要在调整对冲头寸前运行
        # 交易日
        if self.is_trade_day():
            self.hedge()  # 计算对冲手数，需要用当天和历史的数据进行对比，所以应当在数据缓存前进行

    def profit_rolling(self, ):
        """每日滚动计算各项损益"""
        on_trade_day = self.is_trade_day()
        # 资金成本
        borrow_rate = self.cur["npv"]["银行间隔夜利率（%）"]
        self.profile["borrow_interest"].append(borrow_rate / 100 / 365 * self.borrow_cash)
        # 票面利息
        bond_interest = sum(self.bond_info.loc[bond]["每日票面+返税"] / 100 * v for bond, v in self.position.bond)
        self.profile["bond_interest"].append(bond_interest)
        # 净价损益, 非交易日，cur=pre 净价变动为0；交易日，正常
        bond_profit = 0
        cur = self.cur["npv"] if self.cur["npv"] else self.pre_trade_day["npv"]
        pre = self.pre_trade_day["npv"]
        if on_trade_day:
            for bond, v in self.position.bond:
                bond_profit += v * (cur[bond] - pre[bond]) / 100
        self.profile["bond_profit"].append(bond_profit)
        # 期货损益，昨仓*(昨价-今价)*合约乘数，正数是收益
        future_profit = 0
        cur = self.cur["future"]
        pre = self.pre_trade_day["future"]
        if on_trade_day:
            for f, v in self.position.future:
                future_profit += v * 1e6 * (pre[f"{f}年期货收盘价"] - cur[f"{f}年期货收盘价"]) / 100
        self.profile["fut_profit"].append(future_profit)

        self.position.cache_future()
        self.profile["息差"].append(self.profile["bond_interest"][-1] - self.profile["borrow_interest"][-1])
        self.profile["无套期收益"].append(self.profile["息差"][-1] + self.profile["bond_profit"][-1])
        self.profile["套期收益"].append(self.profile["无套期收益"][-1] + self.profile["fut_profit"][-1])
        if self.verbose:
            print(self.today.date(),
                  "票息返税:{}".format("%.2f" % self.profile["bond_interest"][-1], ),
                  "资金成本:{}".format("%.2f" % self.profile["borrow_interest"][-1]),
                  "净价损益:{}".format("%.2f" % self.profile["bond_profit"][-1]),
                  "期货损益:{}".format("%.2f" % self.profile["fut_profit"][-1]), )

    def hedge(self, ):
        """确定套期保值比例、策略，返回合约手数，基本计算方法为dv01加权"""
        self.model()

        future = defaultdict(float)
        for bond, v in self.position.bond:
            bond_dv01 = self.cur["dv"][bond]
            dv01_10 = self.cur["future"][f"{10}年CTD-DV"]
            dv01_5 = self.cur["future"][f"{5}年CTD-DV"]
            dv01_2 = self.cur["future"][f"{2}年CTD-DV"]

            if bond_dv01 >= dv01_10:
                fut_ratio = {10: bond_dv01 / dv01_10, 5: 0, 2: 0}
            elif bond_dv01 >= dv01_5:
                diff = dv01_10 - dv01_5
                fut_ratio = {10: (bond_dv01 - dv01_5) / diff, 5: (dv01_10 - bond_dv01) / diff, 2: 0}
            elif bond_dv01 >= dv01_2:
                diff = dv01_5 - dv01_2
                fut_ratio = {10: 0, 5: (bond_dv01 - dv01_2) / diff, 2: (dv01_5 - bond_dv01) / diff}
            else:
                fut_ratio = {10: 0, 5: 0, 2: 0}
            for f, r in fut_ratio.items():
                h = self.cur["dv"][bond] * v * self.cur["npv"][bond] / 100  # 用净价会好一点
                h *= self.cur["future"][f"{f}年转换因子"] / self.cur["future"][f"{f}年CTD-DV"]
                h /= 1e6 * self.cur["future"][f"{f}年期货收盘价"] / 100
                h *= self.signal_ensemble(bond, f)  # 最终的信号，可以是bool可以是浮点数
                future[f] += h * r
        self.position.update_future({f: round(v) for f, v in future.items()})

    def model(self, ):
        for bond, _ in self.position.bond:
            for f in (10, 5, 2):
                train = {f"{f}_break": self.future_cache[f"{f}年期货收盘价_break_5"][-self.batch_size:],
                         f"{f}_irr_cover": self.future_cache[f"{f}_irr_cover"][-self.batch_size:],
                         f"{bond}_break": self.npv_cache[f"{bond}_break_5"][-self.batch_size:]}
                train = pd.DataFrame(train)
                gt = self.future_cache[f"{f}年期货收盘价"][-self.batch_size:].to_list() + [
                    self.cur["future"][f"{f}年期货收盘价"]]
                gt = np.array(gt[1:]) < np.array(gt[:-1])
                if train.empty or gt.shape[0] == 0 or all(gt) or all(gt ^ 1):
                    return
                if len(train) < self.batch_size:
                    self.model_zoo["LR"][bond][f].fit(train, gt)
                    self.model_zoo["NB"][bond][f].fit(train, gt)
                    self.model_zoo["SVC"][bond][f].fit(train, gt)
                    self.model_zoo["Tree"][bond][f].fit(train, gt)
                else:
                    self.model_zoo["LR"][bond][f].fit(train, gt)
                    self.model_zoo["NB"][bond][f].partial_fit(train, gt)
                    self.model_zoo["SVC"][bond][f].fit(train, gt)
                    self.model_zoo["Tree"][bond][f].fit(train, gt)

                eval = pd.DataFrame({f"{f}_break": [self.cur["future"][f"{f}年期货收盘价_break_5"]],
                                     f"{f}_irr_cover": [self.cur["future"][f"{f}_irr_cover"]],
                                     f"{bond}_break": [self.cur["npv"][f"{bond}_break_5"], ]})
                self.model_hedge["LR"][bond][f] = self.model_zoo["LR"][bond][f].predict_proba(eval)[0][0]
                self.model_hedge["NB"][bond][f] = self.model_zoo["NB"][bond][f].predict_proba(eval)[0][0]

    def signal_ensemble(self, bond, future):
        bond_break = (not self.cur["npv"][f"{bond}_break_5"]) | self.inhibitor["bond_break"]
        future_break = (not self.cur["future"][f"{future}年期货收盘价_break_5"]) | self.inhibitor["future_break"]
        irr_cover = self.cur["future"][f"{future}_irr_cover"] | self.inhibitor["irr_cover"]
        # 约束的条件用 or True 放松约束
        constrain = True
        constrain &= bond_break
        constrain &= (future_break | irr_cover)
        constrain &= (self.model_hedge["NB"][bond][future] > 0.4) | self.inhibitor["NB"]
        constrain &= (self.model_hedge["LR"][bond][future] > 0.4) | self.inhibitor["LR"]
        # 放松的条件用 and False 取消放开
        loosen = False
        loosen |= (self.model_hedge["NB"][bond][future] > 0.8) & (not self.inhibitor["NB"])
        loosen |= (self.model_hedge["LR"][bond][future] > 0.8) & (not self.inhibitor["LR"])
        return constrain | loosen

    def post_process(self, start_date, end_date):
        self.fut_position_his = self.position.get_future_history(start_date, end_date)

        self.profile["累计息差"] = np.cumsum(self.profile["息差"])
        self.profile["累计无套期收益"] = np.cumsum(self.profile["无套期收益"])
        self.profile["累计套期收益"] = np.cumsum(self.profile["套期收益"])
