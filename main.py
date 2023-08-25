"""
这版本起不再计算全价
"""
import datetime

from analyze import analysis, price_plot, future_plot, pnl_plot, plot
from strategy2 import Strategy
from utils import data_prep

dv, nv, fut, code, code_dict, bond_info, ctd_info = data_prep()
bond_info = bond_info.set_index("简称")
ctd_info = ctd_info.set_index("代码")

start_date = datetime.datetime.strptime("2019-01-01", "%Y-%m-%d")
end_date = datetime.datetime.strptime("2023-06-30", "%Y-%m-%d")
back_test_period = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]

if __name__ == '__main__':
    # python main.py
    # 循环自然日
    s = Strategy()
    for dt in back_test_period:
        s.handle_data(dt)
    s.post_process(start_date, end_date)

    print("###当前策略")
    analysis(s, back_test_period, show_baseline=True)
    plot(s, period=back_test_period, fut=fut, nv=nv, plot_contents=("profit", "fut_position"))

    # sign = input("运行消融实验:[yes/no]")
    # if sign.lower().startswith("n"):
    #     sys.exit(0)

    ablation = {"当前策略": s}
    inhibitors = {
        "无择时被动套期": {"bond_break": True, "future_break": True, "irr_cover": True, "NB": True, "LR": True},
        "无突破择时套期": {"bond_break": True, "future_break": True},
        "无irr择时套期": {"irr_cover": True},
        # "无NB": {"NB": True, "LR": False},
        # "无LR": {"NB": False, "LR": True},
        "无模型": {"NB": True, "LR": True},
    }
    pnl = {"当前策略": s.profile["累计套期收益"]}
    for k, v in inhibitors.items():
        tmp = Strategy(inhibitor=v)
        for dt in back_test_period:
            tmp.handle_data(dt)
        tmp.post_process(start_date, end_date)
        print(f"### {k}")
        analysis(tmp, back_test_period, show_baseline=False)
        # plot(tmp, period=back_test_period, fut=fut, nv=nv, plot_contents=("profit", "fut_position"))
        pnl[k] = tmp.profile["累计套期收益"]
    print("###策略对比")
    pnl_plot(back_test_period, s.borrow_cash, **pnl)

    print("###现券价格和国债期货价格走势")
    price_plot("18北京债05", bond_info, nv, fut)
    price_plot("15广东债18", bond_info, nv, fut)
    print("###国债收益率、期货ytm, irr、拆借利率")
    future_plot(s, fut, f=10)
    future_plot(s, fut, f=5)
    future_plot(s, fut, f=2)
