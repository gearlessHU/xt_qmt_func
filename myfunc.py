__author__ = 'Willows'
__version__ = "1205"


import pandas as pd
import numpy as np
import os
import re
import datetime
from xtquant import xtdata
import pickle
from typing import Union, List, Any, Optional, Coroutine, Callable, Tuple, Dict
import logging
from collections import deque
import json
import requests

# --------------------log记录程序---------------------------
log_file_path = ""
log_file_name = ""
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s %(levelname)s: %(message)s',  # 设置日志格式
    datefmt = '%Y-%m-%d %H:%M:%S',  # 设置日期时间格式
    filename = os.path.join(log_file_path,log_file_name + ".log"),  # 指定日志文件路径
    filemode = 'w'  # 设置写入模式，'w'表示覆盖原有日志，'a'表示追加到原有日志末尾
)


# --------------------股票版本------------------------------
def my_passorder(C,stock,opentype,lots,price = None,m_strRemark = '系统备注'):
    '''
    
    Args:
        C: ContextInfo
        stock: 股票代码
        
        opentype: 
                'buy' 买股票
                'sell' 卖股票
        lots: 股数
        price: 下单价格，不指定时默认按对手价下单
        m_strRemark = '系统备注' 用于自定义寻找orderID
    '''
    #holdings = get_holdings(C.accID,'stock')
    opentype = opentype #买卖方向
    op =1101  #股数
    opType = 23 if opentype == 'buy' else 24# 若不为'sell' 则为buy
    volumex = lots
    price = 0 if not price else price # price参数必须存在
    prType = 14 if not price else 11 # 若不指定价格，则默认按对手价下单
    print(f'{stock} 新委托信息 方向{opentype} 价格{price} 量{volumex}')
    #print(f"opType:{opType} , op:{op} , C.accID{C.accID} , stock{stock} , prType{prType} , price{price} , volumex{volumex}")
    passorder(opType, op, C.accID,stock, prType, price, volumex,'交易注释',1,'{}'.format(m_strRemark), C)
    print(f'委托发送完成')


# --------------------股票版本，防超单版本，需要配合refresh_waiting_dict使用------------------------------

def refresh_waiting_dict(accID):
    if "waiting_dict" not in globals():
        global waiting_dict
        waiting_dict = {}
    #获取委托信息
    order_list = get_trade_detail_data(accID,"STOCK","order")
    #取出委托对象的 投资备注 : 委托状态
    ref_dict = {i.m_strRemark : i.m_nOrderStatus for i in order_list}
    
    # print(ref_dict)
    del_list = []
    
    for stock in waiting_dict:
        # print(ref_dict,ref_dict[A.waiting_dict[stock]])
        if waiting_dict[stock] in ref_dict and ref_dict[waiting_dict[stock]] in [50, 53, 54, 56, 57, 55]:
            print(f'查到投资备注 {waiting_dict[stock]}，的委托 状态{ref_dict[waiting_dict[stock]]} (56已成 53部撤 54已撤)从等待等待字典中删除')
            del_list.append(stock)
    for stock in del_list:
        del waiting_dict[stock]


def my_passorder_V2(C,stock,opentype,lots,price = None,m_strRemark = ""):
    '''
    
    ToDO: 需要配合refresh_waiting_dict()使用

    Args:
        C: ContextInfo
        stock: 股票代码
        
        opentype: 
                'buy' 买股票
                'sell' 卖股票
        lots: 股数
        price: 下单价格，不指定时默认按对手价下单
        m_strRemark = '系统备注' 用于自定义寻找orderID
    '''
    dateNow = datetime.datetime.now()
    dateNow = dateNow.hour * 3600 + dateNow.minute * 60 + dateNow.second
    # 交易前检查
    if "waiting_dict" not in globals():
        global waiting_dict
        waiting_dict = {}

    if stock in waiting_dict:
        print(f"{stock} 未查到或存在未撤回委托 {waiting_dict[stock]} 暂停后续报单")
        return
    
    if m_strRemark == "":
        m_strRemark = f"{dateNow}_{stock}_系统备注"

    #holdings = get_holdings(C.accID,'stock')
    opentype = opentype #买卖方向
    op =1102 if opentype == "buy" else 1101 #股数
    opType = 23 if opentype == 'buy' else 24# 若不为'sell' 则为buy
    volumex = lots
    price = 0 if not price else price # price参数必须存在
    prType = 14 if not price else 11 # 若不指定价格，则默认按对手价下单
    print(f'{stock} 新委托信息 方向{opentype} 价格{price} 量{volumex}')
    #print(f"opType:{opType} , op:{op} , C.accID{C.accID} , stock{stock} , prType{prType} , price{price} , volumex{volumex}")
    passorder(opType, op, C.accID,stock, prType, price, volumex,'交易注释',1,m_strRemark, C)
    waiting_dict[stock] = m_strRemark
    print(f'委托发送完成')



#------------------期货版本--------------------------
def my_passorder(C,Future:str,opentype:str,lots:int,price = None,m_strRemark = '系统备注'):
    '''
    
    Args:
        C: ContextInfo \n
        Future: 期货代码 \n
        
        opentype: 
                'buy_open' 开多\n
                'sell_open' 开空\n
                'sell_close' 平多\n
                'buy_close' 平空\n
        lots: 手
        price: 下单价格，不指定时默认按市价下单
        m_strRemark = '系统备注' 用于自定义寻找orderID
    '''
    Future_ExchangeID = Future.split(".")[1]
    opentype = opentype #买卖方向
    op =1101  #手数
    # 期货区分开平
    if opentype == "buy_open":
        opType = 0
    elif opentype == "sell_open":
        opType = 3
    elif opentype == "sell_close":
        opType = 7
    elif opentype == "buy_close":
        opType = 9

    volumex = lots
    price = 0 if not price else price # price参数必须存在
    if Future_ExchangeID == "SF":
        prType = 14 if not price else 11 # 对于上期所，若不指定价格，则默认按对手价下单
    elif Future_ExchangeID == "DF" or Future_ExchangeID == "ZF":
        prType = 12 if not price else 11 # 对于大商所和郑商所，若不指定价格，则默认按涨跌停价下单
    else:
        prType = 14 if not price else 11 # 对于其他所，若不指定价格，则默认按对手价下单

    print(f'{Future} 新委托信息 方向{opentype} 价格{price} 量{volumex}')
    #print(f"opType:{opType} , op:{op} , C.accID{C.accID} , stock{stock} , prType{prType} , price{price} , volumex{volumex}")
    passorder(opType, op, C.accID,Future, prType, price, volumex,'交易注释',1,'{}'.format(m_strRemark), C)
    print(f'委托发送完成')


def get_stock_holdings(accid,symbol = None):
    '''
    Arg:
        accondid:账户id
        datatype:'FUTURE'：期货,'STOCK'：股票,......
        symbol: 品种，不填默认返会全部持仓
            
    return:
        {股票代码:{'手数':int,"持仓成本":float,'浮动盈亏':float,"可用余额":int}}
    '''
    PositionInfo_dict={}
    resultlist = get_trade_detail_data(accid,"STOCK",'POSITION')
    for obj in resultlist:
        PositionInfo_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID] = {
        "持仓数量":obj.m_nVolume,
        "持仓成本":obj.m_dOpenPrice,
        "浮动盈亏":obj.m_dFloatProfit,
        "盈亏比例":obj.m_dProfitRate,
        "市值":obj.m_dMarketValue,
        "可用余额":obj.m_nCanUseVolume,
        "交易方向":obj.m_nDirection
        }
    if symbol:
        return PositionInfo_dict[symbol]
    else :
        return PositionInfo_dict
    
def get_holding_v2(accID,datatype,order,symbol = None):
    '''
    ToDo: dbf用获取持仓

    Args: 
        accID : 账号;
        datatype: 市场["STOCK","FUTURE".....]
        order: dbf接口,需要在策略中手动指定
    '''
    
    PositionInfo_dict={}
    resultlist = order.get_trade_detail_data(accID,datatype,'POSITION')
    for obj in resultlist:
        PositionInfo_dict[obj['证券代码'] + "." + obj["市场代码"]] = {
            '证券名称' : obj['证券名称'],
            '买卖' : obj['买卖'],
            '持仓成本': obj['持仓成本'],
            '成本价': obj['成本价'],
            '盈亏' : obj['盈亏'],
            '市值' : obj['市值'],
            '可用数量' : obj['可用数量'],
            '当前拥股' : obj['当前拥股']
        }
    if symbol:
        return PositionInfo_dict[symbol]
    else:
        return PositionInfo_dict


def get_Future_holdings(accid,symbol = None):
    '''
    针对期货返回持仓的奇葩结构做处理
    Arg:
        accondid:账户id
        symbol: 品种，不填默认返会全部持仓
            
    return:
        {股票名:{'手数':int,"持仓成本":float,'浮动盈亏':float,"可用余额":int}}
    '''
    datatype = "FUTURE"
    
    PositionInfo_dict = {}
    
    Long_dict={}
    
    Short_dict={}
    
    resultlist = get_trade_detail_data(accid,datatype,'POSITION')
    
    for obj in resultlist:
        #防除零
        if obj.m_nVolume == 0:
            continue
        if obj.m_nDirection == 48:
            if not Long_dict.get(obj.m_strInstrumentID+"."+obj.m_strExchangeID):
                Long_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID] = {
                "多头数量":obj.m_nVolume,
                "多头成本":obj.m_dOpenPrice,
                
                "浮动盈亏":obj.m_dFloatProfit,
                "保证金占用":obj.m_dMargin
                }
            else:
                    
                    Long_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID]["多头数量"] += obj.m_nVolume
                    # 算浮动盈亏
                    Long_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID]["浮动盈亏"] += obj.m_dFloatProfit
                    # 算保证金占用
                    Long_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID]["保证金占用"] += obj.m_dMargin
                    # 算多头成本
                    Long_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID]["多头成本"] = (
                        Long_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID]["多头成本"] * \
                        (Long_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID]["多头数量"] - obj.m_nVolume) + \
                        (obj.m_dOpenPrice * obj.m_nVolume)
                    )/Long_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID]["多头数量"]
                    
        elif obj.m_nDirection == 49:
            if not Short_dict.get(obj.m_strInstrumentID+"."+obj.m_strExchangeID):
                Short_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID] = {
                "空头数量":obj.m_nVolume  ,
                "空头成本":obj.m_dOpenPrice ,
                "浮动盈亏":obj.m_dFloatProfit,
                "保证金占用":obj.m_dMargin
                }
            else:
                Short_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID]["空头数量"] += obj.m_nVolume
                # 算浮动盈亏
                Short_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID]["浮动盈亏"] += obj.m_dFloatProfit
                # 算保证金占用
                Short_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID]["保证金占用"] += obj.m_dMargin
                # 计算空头成本
                Short_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID]["空头成本"] = (
                    Short_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID]["空头成本"] * \
                    (Short_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID]["空头数量"] - obj.m_nVolume) + \
                    (obj.m_dOpenPrice * obj.m_nVolume)
                )/Short_dict[obj.m_strInstrumentID+"."+obj.m_strExchangeID]["空头数量"]
        
    
    for _symbol in set(list(Long_dict.keys()) + list(Short_dict.keys())):
        
        PositionInfo_dict[_symbol] = {
        "多头数量":Long_dict[_symbol]["多头数量"] if Long_dict.get(_symbol) else 0 ,
        
        "空头数量":Short_dict[_symbol]["空头数量"] if Short_dict.get(_symbol) else 0 ,
        
        "多头成本":Long_dict[_symbol]["多头成本"] if Long_dict.get(_symbol) else None ,
        
        "空头成本":Short_dict[_symbol]["空头成本"] if Short_dict.get(_symbol) else None,
        
        "净持仓" : Long_dict.get(_symbol,{}).get("多头数量",0) -  Short_dict.get(_symbol,{}).get("空头数量",0),
        
        "浮动盈亏": Long_dict.get(_symbol,{}).get("浮动盈亏",0) +  Short_dict.get(_symbol,{}).get("浮动盈亏",0),
        
        "保证金占用": Long_dict.get(_symbol,{}).get("保证金占用",0) +  Short_dict.get(_symbol,{}).get("保证金占用",0)
        }
        
    if symbol:
        return PositionInfo_dict[symbol]
    else :
        return PositionInfo_dict
    

def get_trade(accid,datatype,symbol = None):
    '''
    Arg:
        accondid:账户id
        datatype:'FUTURE'：期货,'STOCK'：股票,......
        symbol: 品种，不填默认返会全部持仓
            
    return:
        {股票名:{'手数':int,"成交编号":float,'成交方向':float,"成交均价":int, "成交量":int, "成交类型":int, "操作类型":int, "手续费":float, "成交额":float}}
    '''
    TradeInfo_dict = {}
    resultlist = get_trade_detail_data(accid,datatype,"DEAL")
    for obj in resultlist:
        TradeInfo_dict[obj.m_strInstrumentID + "." + obj.m_strExchangeID] = {
        "成交编号":obj.strTradeID,
        "成交方向":obj.m_nOffsetFlag,
        "成交均价":obj.m_dPrice,
        "成交量":obj.m_nVolume,
        "成交类型":obj.m_eFutureTradeType,
        "操作类型":obj.m_nRealOffsetFlag,
        "手续费":obj.m_dComssion,
        "成交额":obj.m_dTradeAmount,
        }
    if symbol:
        return TradeInfo_dict[symbol]
    else:
        return TradeInfo_dict


def get_account(accid,datatype):
    Account_dict = {}
    resultlist = get_trade_detail_data(accid,datatype,"ACCOUNT")
    for obj in resultlist:
        Account_dict[obj.m_strAccountID] = {
            "冻结金额":obj.m_dFrozenCash,
            "总资产":obj.m_dBalance,
            "可用金额":obj.m_dAvailable,
            "手续费":obj.m_dCommission,
            "持仓盈亏":obj.m_dPositionProfit
        }


def get_order(accID, datatype, symbol = None):
    OrderInfo_dict = {}
    resultlist = get_trade_detail_data(accid,datatype,"ORDER")
    for obj in resultlist:
        OrderInfo_dict[obj.m_strInstrumentID + "." + obj.m_strExchangeID] = {
            "委托量":obj.m_nVolumeTotalOriginal,
            "合同编号":obj.m_strOrderSysID,
            "委托价格":obj.m_dLimitPrice,
            "委托状态":obj.m_nOrderStatus,
            "成交数量": obj.m_nVolumeTraded,
            "委托剩余量":obj.m_nVolumeTotal,
            "委托日期":obj.m_strInsertDate,
            "委托时间":obj.m_strInsertTime,
            "多空方向":obj.m_nDirection,
            "开平":obj.m_nOffsetFlag,
            "投资备注":obj.m_strRemark
        }
    if symbol:
        return OrderInfo_dict[symbol]
    else:
        return OrderInfo_dict


def get_order_v2(accID,datatype, order, symbol = None):
    '''
    ToDo: dbf用获取委托

    Args: 
        accID : 账号;
        datatype: 市场["STOCK","FUTURE".....]
        order: dbf接口,需要在策略中手动指定
    '''
    OrderInfo_dict = {}
    resultlist = order.get_trade_detail_data(accID,datatype,"ORDER")
    for obj in resultlist:
        if obj["交易市场"] == "深交所":
            strMarket = "SZ"
        else:
            strMarket = "SH"
        OrderInfo_dict[obj['证券代码'] + "." + strMarket] = {
            "委托量":obj["委托量"],
            "合同编号":obj["合同编号"],
            "委托价格":obj["委托价格"],
            "委托状态":obj["委托状态"],
            "成交数量": obj["成交数量"],
            "委托剩余量":obj["委托剩余量"],
            "委托日期":obj["委托日期"],
            "委托时间":obj["委托时间"],
            "开平":obj["开平"],
            "投资备注":obj["投资备注"]
            }
    if symbol:
        return OrderInfo_dict[symbol]
    else:
        return OrderInfo_dict




def llv(series,n):
    '''
    求在n个周期内的最小值
    注意: n为0的情况下, 或当n为有效值但当前的series序列元素个数不足n个, 函数返回 NaN 序列
    Args:
        series:数据序列
        n:周期
    Return:
        series:数据序列
    '''
    llv_data = series.rolling(n).min()
    return llv_data

# 实现通达信LLV功能，在数据不足的情况下使用部分有效值进行计算
def optimized_rolling_min(series:pd.Series, window:int): 
    values = series.to_list()
    queue = deque()
    results = []

    for i, item in enumerate(values):
        if len(queue) == window:
            queue.popleft()
        
        # Only append non-NaN values to the queue
        if pd.notnull(item):
            queue.append(item)

        # Calculate max_value only when queue is not empty
        if queue: 
            results.append(min(queue))
        else:
            results.append(np.nan)  # Append NaN if the queue is empty

    return pd.Series(results,index = series.index)



def hhv(series:pd.Series,n):
    '''
    求在n个周期内的最大值
    注意: n为0的情况下, 或当n为有效值但当前的series序列元素个数不足n个, 函数返回 NaN 序列
    Args:
        series:数据序列
        n:周期
    Return:
        series:数据序列
    '''
    hhv_data = series.rolling(n).max()
    return hhv_data

# 实现通达信HHV功能，在数据不足的情况下使用部分有效值进行计算
def optimized_rolling_max(series:pd.Series, window:int):
    values = series.to_list()
    queue = deque()
    results = []

    for i, item in enumerate(values):
        if len(queue) == window:
            queue.popleft()
        
        # Only append non-NaN values to the queue
        if pd.notnull(item):
            queue.append(item)

        # Calculate max_value only when queue is not empty
        if queue: 
            results.append(max(queue))
        else:
            results.append(np.nan)  # Append NaN if the queue is empty

    return pd.Series(results,index = series.index)





def ATR(c_df:pd.DataFrame,h_df:pd.DataFrame,l_df:pd.DataFrame,N:int):
    # 特殊数据格式计算atr

    h_c_diff = h_df - l_df #
    pre_c_h_diff = (c_df.shift(1) - h_df).abs()
    pre_c_l_diff = (c_df.shift(1) - l_df).abs()

    tr_zb1 = h_c_diff * (h_c_diff >= pre_c_h_diff)
    tr_zb2 = tr_zb1 + pre_c_h_diff * (pre_c_h_diff > h_c_diff) # 得到了zb1和zb2中最大值的表

    tr_zb3 = tr_zb2 * (tr_zb2 >= pre_c_l_diff)
    tr = tr_zb3 + pre_c_l_diff * (pre_c_l_diff > tr_zb2) 

    _Atr = tr.apply(lambda x: x.rolling(N).mean().round(2))

    return _Atr


# 计算df中每个值的历史分位数
def hist_quantiles(df:pd.DataFrame):
    _n = pd.DataFrame(0,index=df.index,columns=df.columns)
    for col in df.columns:
        hist_quantiles = df[col].expanding().apply(lambda x: x.rank(pct = True).iloc[-1])
        #print(hist_quantiles)
        # 将历史分位数加入df中
        _n[col] = hist_quantiles
    return _n


def cointegration_check(series01, series02):
    '''
    To Do: 协整性检验
    
    Arg:
        series01: pd.Series
        series02: pd.Series
    
    '''
    '''需要3.8以上的版本'''
    urt_1 = ts.adfuller(np.array(series01), 1)[1]
    urt_2 = ts.adfuller(np.array(series02), 1)[1]

    # 同时平稳或不平稳则差分再次检验
    if (urt_1 > 0.1 and urt_2 > 0.1) or (urt_1 < 0.1 and urt_2 < 0.1):
        urt_diff_1 = ts.adfuller(np.diff(np.array(series01)), 1)[1]
        urt_diff_2 = ts.adfuller(np.diff(np.array(series02), 1))[1]

        # 同时差分平稳进行OLS回归的残差平稳检验
        if urt_diff_1 < 0.1 and urt_diff_2 < 0.1:
            matrix = np.vstack([series02, np.ones(len(series02))]).T
            beta, c = np.linalg.lstsq(matrix, series01, rcond=None)[0]
            resid = series01 - beta * series02 - c
            if ts.adfuller(np.array(resid), 1)[1] > 0.1:
                result = False
            else:
                result = True
            return beta, c, resid, result
        else:
            result = False
            return 0.0, 0.0, 0.0, result

    else:
        result = False
        return 0.0, 0.0, 0.0, result

def ATR(df, n):
    """
    平均真实波幅

    Args:
        df (pandas.DataFrame): Dataframe格式的K线序列

        n (int): 平均真实波幅的周期

    Returns:
        pandas.DataFrame: 返回的DataFrame包含2列, 分别是"tr"和"atr", 分别代表真实波幅和平均真实波幅

    Example: 
        atr = ATR(klines, 14)
        print(atr.tr)  # 真实波幅
        print(atr.atr)  # 平均真实波幅

        # 预计的输出是这样的:
        [..., 143.0, 48.0, 80.0, ...]
        [..., 95.20000000000005, 92.0571428571429, 95.21428571428575, ...]
    """
    new_df = pd.DataFrame()
    pre_close = df["close"].shift(1)
    new_df["tr"] = np.where(df["high"] - df["low"] > np.absolute(pre_close - df["high"]),
                            np.where(df["high"] - df["low"] > np.absolute(pre_close - df["low"]),
                                     df["high"] - df["low"], np.absolute(pre_close - df["low"])),
                            np.where(np.absolute(pre_close - df["high"]) > np.absolute(pre_close - df["low"]),
                                     np.absolute(pre_close - df["high"]), np.absolute(pre_close - df["low"])))

    new_df["atr"] = new_df["tr"].rolling(n).mean()
    return new_df


def MA(S,N):           #求序列的N日平均值，返回序列                    
    return pd.Series(S).rolling(N).mean()

def SMA(S, N, M=1):   #麦语言式的SMA,至少需要120周期才精确     
    K = pd.Series(S).rolling(N).mean()    #先求出平均值 (下面如果有不用循环的办法，能提高性能，望告知)
    for i in range(N+1, len(S)):  K[i] = (M * S[i] + (N - M) * K[i-1]) / N  # 因为要取K[i-1]，所以 range(N+1, len(S))        
    return K

def sma(series, n, m):
    """
    扩展指数加权移动平均: 求series序列n周期的扩展指数加权移动平均
    
        计算公式:
        sma(x, n, m) = sma(x, n, m).shift(1) * (n - m) / n + x(n) * m / n
        
        注意: n必须大于m

    Args:
        series (pandas.Series): 数据序列
        
        n (int): 周期
        
        m (int): 权重

    Returns:
        pandas.Series: 扩展指数加权移动平均序列
    """
    sma_data = series.ewm(alpha=m / n, adjust=False).mean()
    return sma_data


def COUNT(S, N):                  # COUNT(CLOSE>O, N):  最近N天满足S_BOO的天数  True的天数
    return pd.Series(S).rolling(N).sum() 

def LLV(S,N):                           # LLV(C, 5)  # 最近5天收盘最低价     
    return pd.Series(S).rolling(N).min()



def REF(S, N=1):       #对序列整体下移动N,返回序列(shift后会产生NAN)    
    return pd.Series(S).shift(N)  
    
def EMA(S,N):         #指数移动平均,为了精度 S>4*N  EMA至少需要120周期       
    return pd.Series(S).ewm(span=N, adjust=False).mean()    
    
def ema(df:pd.DataFrame,N):                
    return df.ewm(span=N, adjust=False).mean()  


def KDJ(close:pd.DataFrame, high:pd.DataFrame, low:pd.DataFrame, N = 9, M1 = 3, M2 = 3):
    RSV = (close - low.rolling(N).min()) / (high.rolling(N).max() - low.rolling(N).min()) * 100
    K = RSV.apply(lambda x: EMA(x,M1*2 - 1))
    D = K.apply(lambda x: EMA(x,M2*2 - 1))
    J=K*3-D*2
    return K, D, J

def KDJ(close:pd.DataFrame, high:pd.DataFrame, low:pd.DataFrame, N = 9, M1 = 3, M2 = 3):
    RSV = (close - low.apply(lambda x: optimized_rolling_min(x,N))) / (high.apply(lambda x:optimized_rolling_max(x,N)) - low.apply(lambda x: optimized_rolling_min(x,N))) * 100
    # K = RSV.apply(lambda x: EMA(x,M1*2 - 1)).round(3)
    # D = K.apply(lambda x: EMA(x,M2*2 - 1)).round(3)
    K = RSV.apply(lambda x: sma(x,M1 ,1)).round(3)
    D = K.apply(lambda x: sma(x,M2 , 1)).round(3)
    J=K*3-D*2
    
    return K, D, J


def qrr_rate(volume_df:pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        volume_df:标准格式volume
    return:
        pd.DataFrame:日线级别的量比数据
    """
    volume_1m_mean_df = (volume_df / 240).round(2)
    volume_pre5DaysMean_df = (volume_df.shift(1).rolling(5).sum()) / (240 * 5)
    day_qrr_rate_df = (volume_1m_mean_df / volume_pre5DaysMean_df).round(2)
    day_qrr_rate = day_qrr_rate_df.iloc[-1]

def CROSSUP(a, b):
    """
    向上穿越: 表当a从下方向上穿过b, 成立返回1, 否则返回0
    Args:
        a (pandas.Series): 数据序列1

        b (pandas.Series): 数据序列2

    Returns:
        pandas.Series: 上穿标志序列
    """
    crossup_data = pd.Series(np.where((a > b) & (a.shift(1) <= b.shift(1)), 1, 0))
    return crossup_data

def CROSSDOWN(a, b):
    """
    向下穿越: 表示当a从上方向下穿b，成立返回1, 否则返回0
    Args:
        a (pandas.Series): 数据序列1

        b (pandas.Series): 数据序列2

    Returns:
        pandas.Series: 下穿标志序列
    """
    crossdown_data = pd.Series(np.where((a < b) & (a.shift(1) >= b.shift(1)), 1, 0))
    return crossdown_data


def FILTER(S,N):                       #信号过滤
    temp=-1
    l=[]
    for x,v in S.items():
        if v:
            if temp==-1 or x-temp>N:
                l.append(True)
                temp=x
            else:
                l.append(False)
        else:
            l.append(False)
    return pd.Series(l)


def barlast(cond:pd.Series) -> np.array:
    """
    返回一个序列，其中每个值表示从上一次条件成立到当前的周期数

    (注： 如果从cond序列第一个值到某个位置之间没有True，则此位置的返回值为 -1； 条件成立的位置上的返回值为0)


    Args:
        cond (pandas.Series): 条件序列(序列中的值需为 True 或 False)

    Returns:
        np.array : 周期数序列（其长度和 cond 相同；最后一个值即为最后一次条件成立到最新一个数据的周期数）


    """
    cond = cond.to_numpy()
    v = np.array(~cond, dtype=np.int64)
    c = np.cumsum(v)
    x = c[cond]
    d = np.diff(np.concatenate(([0], x)))
    if len(d) == 0:  # 如果cond长度为0或无True
        return pd.Series([-1] * len(cond))
    v[cond] = -d
    r = np.cumsum(v)
    r[:x[0]] = -1
    return r


def ADTM(open_df:pd.DataFrame,high_df:pd.DataFrame,low_df:pd.DataFrame,p:int,n:int):
    dtm_zb1 = open_df > open_df.shift(1)
    dtm_zb2 = high_df - open_df
    dtm_zb3 = open_df - open_df.shift(1)
    dtm_max = dtm_zb2 * (dtm_zb2 >= dtm_zb3)
    dtm_max = dtm_max + dtm_zb3 * (dtm_zb3 > dtm_zb2)
    dtm = dtm_max * dtm_zb1

    dbm_zb1 = open_df < open_df.shift(1)
    dbm_zb2 = open_df - low_df
    dbm_zb3 = open_df - open_df.shift(1)
    dbm_max = dbm_zb2 * (dtm_zb2 >= dbm_zb3)
    dbm_max = dbm_max + dbm_zb3 * (dbm_zb3 > dbm_zb2)
    dbm = dbm_max * dbm_zb1

    stm = dtm.rolling(p).sum().round(2)
    sbm = dbm.rolling(p).sum().round(2)

    adtm_zb1 = ~(stm - sbm == 0) * ((stm - sbm) / sbm)
    adtm_zb1.replace(np.inf, np.nan, inplace=True)

    adtm_zb2 = adtm_zb1 * (stm < sbm) + ((stm - sbm) / stm * (stm > sbm))
    _adtm = adtm_zb2.round(2)

    _ma1 = _adtm.rolling(n).mean().round(2)
    

# def calculate_slope(arr):
#     # 创建一个等差数组，表示过去的天数
#     days = np.arange(len(arr))
    
#     # 计算斜率（polyfit函数返回的第一个值为斜率，第二个值为截距）
#     slope = np.polyfit(days, arr, 1)[0]

#     return slope

def calculate_slope(y):
    """
    Args:
        y:要计算的数据序列
    return:
        斜率序列
    """
    # 计算斜率
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return slope


def is_trade_time(trading_time_info):
    '''
    Args:
        trading_time_info:格式需要如下
            C.stock_trade_time = (["09:30:00","11:30:00"],["13:00:00","15:00:00"])
            C.future_trade_time = (["09:00:00","10:15:00"],["10:30:00","11:30:00"],["13:30:00","15:00:00"],["21:00:00","26:30:00"])
    return:bool
    '''
    
    _now = int((datetime.datetime.now() - datetime.timedelta(hours=4)).strftime("%H%M%S"))
    for _time_list_ in trading_time_info:
        st_str = _time_list_[0]
        _sp_st = (int(st_str.split(":")[0]) - 4) * 10000 + (int(st_str.replace(":", "")) % 10000)
        et_str = _time_list_[1]
        _sp_et = (int(et_str.split(":")[0]) - 4) * 10000 + (int(et_str.replace(":", "")) % 10000)
        
        if _sp_st <= _now < _sp_et:
            return True
    return False


def expanding_min_max_normalization(series:pd.DataFrame):
    """
    数据标准化处理
    """
    _min = series.expanding().min()
    _max = series.expanding().max()
    return (series - _min)/(_max-_min)




def get_st_df(df:pd.DataFrame,data = False) -> pd.DataFrame:
    """
    Args:
        df: 任意符合标准框架数据的df,周期为日线
    Return:
        pd.DataFrame: st，*st日的值为False
    """
    df1 = df.copy()
    for i in df1.columns:
        his_st = get_st_status(i)
        for k,v in his_st.items():
            for st_data in v:
                time1 = st_data[0]
                time2 = st_data[1]
                df1.loc[time1:time2,i] = data
    return df1

def get_st_series(s:pd.Series) -> pd.Series:
    """
    Args:
        s: 任意日线索引的Series
    Return:
        pd.Series: st，*st日的值为False的Series
    """
    his_st = get_st_status(s.name)
    for k,v in his_st.items():
        for st_data in v:
            time1 = st_data[0]
            time2 = st_data[1]
            s.loc[time1:time2] = False
    return s
    

def init_data_qmt(C,stock_list:list,period:str,count = -1):
    '''
    Args:
        stock_list:股票代码表
        period: 数据周期
        count: 数据长度
    Return:
        _dict,df 
    '''
    _dict = C.get_market_data_ex_ori([],stock_list,period = period, count = count)
    _dateList = _dict[list(_dict.keys())[0]]["stime"]
    df = pd.DataFrame(index = _dateList , columns = list(_dict.keys()))

    return _dict,df

def init_data_xt(stock_list:list,period:str,count = -1):
    '''
    Args:
        stock_list:股票代码表
        period: 数据周期
        count: 数据长度
    Return: 
        _dict,df
    '''
    _dict = xtdata.get_market_data_ex_ori([],stock_list,period = period, count=count)
    _dateList = _dict[list(_dict.keys())[0]]["stime"]
    df = pd.DataFrame(index = _dateList , columns = list(_dict.keys()))

    return _dict,df


def get_df(dt:dict, df:pd.DataFrame, values_name:str) -> pd.DataFrame:
    '''
    循环从字典里赋值矩阵
    values_name可选字段: ['time', 'stime', 'open', 'high', 'low', 'close', 'volume','amount', 'settelementPrice', 'openInterest', 'preClose', 'suspendFlag']
    '''
    df1 = df.copy()
    df1 = df1.apply(lambda x: dt[x.name][values_name])

    return df1

def get_df_ex(data:dict,field:str) -> pd.DataFrame:
    '''
    ToDo:用于在使用get_market_data_ex的情况下，取到标准df
    
    Args:
        data: get_market_data_ex返回的dict
        field: ['time', 'open', 'high', 'low', 'close', 'volume','amount', 'settelementPrice', 'openInterest', 'preClose', 'suspendFlag']
        
    Return:
        一个以时间为index，标的为columns的df
    
    '''
    _index = data[list(data.keys())[0]].index.tolist()
    _columns = list(data.keys())
    df = pd.DataFrame.from_dict({col: data[col][field] for col in _columns})
    return df



def rank_filter(df:pd.DataFrame, N:int, axis = 1,ascending = False, method = "max", na_option = "keep") -> pd.DataFrame:
    """
    Args:
        df: 标准数据的df
        N: 判断是否是前N名
        axis: 默认是横向排序
        ascending : 默认是降序排序
        na_option : 默认保留nan值,但不参与排名
    Return:
        pd.DataFrame:一个全是bool值的df
    """
    _df = df.copy()
    
    _df = _df.rank(axis = axis, ascending = ascending, method = method, na_option = na_option)
    
    return _df <= N

def get_pre_data(df:pd.DataFrame,time:str,N = 1) -> pd.DataFrame:
    """
    ToDo:获取前N日某一时间点(time)的数据，空数据会用后值填充
    Args:
        df: 标准数据格式
        time: "时分秒"格式字符串，如"150000"
        data_period: 传入数据的周期，可选值{1m}
        N: 向前取N日数据
    return:pd.DataFrame
    """
    new_df = pd.DataFrame(np.nan, index=df.index, columns = df.columns)
    df1 = df.loc[df.index.str[-6:] == time]
    if not df1.shape[0]:
        raise KeyError(f"原数据中可能没有输入的time:{time}")
    pre_data = df1.shift(N)
    new_df.update(pre_data)
    new_df = new_df.fillna(method="bfill")
    return new_df

    
def compare_max(*dataframes:pd.DataFrame) -> pd.DataFrame:
    """
    ToDo: 比较多个df相同位置的最大值
    Args:
        行列完全相同的多个pd.DataFrame
    Return:
        pd.DataFrame
    """

    df_max = pd.DataFrame(np.maximum.reduce([df.values for df in dataframes]),index = dataframes[0].index, columns= dataframes[0].columns)

    return df_max


def compare_min(*dataframes:pd.DataFrame) -> pd.DataFrame:
    """
    ToDo: 比较多个df相同位置的最小值
    Args:
        行列完全相同的多个pd.DataFrame
    Return:
        pd.DataFrame
    """

    df_min = pd.DataFrame(np.minimum.reduce([df.values for df in dataframes]),index = dataframes[0].index, columns= dataframes[0].columns)

    return df_min

def qmt_tocsv(path:str,*args):
    """
    Args:
        path:文件保存的路径
        *args:要保存的变量名
    """
    try:
        for i in [args_name for args_name in args]:
            filename = i + ".csv"
            save_name = os.path.join(path,filename)
            eval(f"{i}.to_csv(r'{save_name}')")
    except Exception as e:
        print("保存失败",e)
        return False
    finally:
        True

def write_pickle(data:Union[pd.DataFrame,dict], path:str ,strategy:str):
    """
    ToDo:写入缓存变量到plk文件中
    Args:
        data: 要存入的数据
        path: 真实存在的本地文件夹
        strategy: 策略的名字
    return: bool,成功则返回True
    """
    try:
        if type(data) == pd.DataFrame:
            data_type = "df"
        elif type(data) == dict:
            data_type = "dict"
        else:
            raise TypeError("传入的参数类型错误，必须是pd.DataFrame，或者dict")
    except TypeError:
        return False
    try:
        with open(os.path.join(path, strategy+ ".pkl"),'wb') as f:
            pickle.dump(data, f)
            return True
    except FileNotFoundError:
        print("文件路径不存在")
        return False


def get_resid(s1,s2):
    '''
    计算残差
    '''
    s1 = sm.add_constant(s1)
    s1 = s1.dropna(how = "any")
    if s1.shape[0] > 0:
        s2 = s2[s1.index]
        model1 = sm.OLS(s2,s1)
        result = model1.fit()
        return result.resid
    else:
        return pd.Series(np.nan,index=s1.index)
    
def get_all_symbol_code(all_code_list) -> list:
    """
    ToDo : 筛选出list里的期货代码

    return: list

    """
    s_dict = {}
    future_list = []
    pattern = r'^[a-zA-Z]{1,2}\d{3,4}\.[A-Z]{2}$'
    # s_dict = {re.findall(r"[a-zA-Z]+",i.split(".")[0])[0] : [] for i in future_list}
    for i in all_code_list:
        
        if re.match(pattern,i):
            future_list.append(i)
    return future_list

def get_option_code(market,data_type = 0):

    '''

    ToDo:取出指定market的期权合约

    Args:
        market: 目标市场，比如中金所填 IF 

    data_type: 返回数据范围，可返回已退市合约，默认仅返回当前

        0: 仅当前
        1: 仅历史
        2: 历史 + 当前
    
    '''
    _history_sector_dict = {
        "IF":"过期中金所",
        "SF":"过期上期所",
        "DF":"过期大商所",
        "ZF":"过期郑商所",
        "GF":"过期广期所",
        "INE":"过期能源中心",
        "SHO":"过期上证期权",
        "SZO":"过期深证期权",
    }

    # _now_secotr_dict = {
    #     "IF":"中金所",
    #     "SF":"上期所",
    #     "DF":"大商所",
    #     "ZF":"郑商所",
    #     "INE":"能源中心",
    #     "SHO":"上证期权",
    #     "SZO":"深证期权",
    # }

    _sector = _history_sector_dict.get(market)
    # _now_sector = _now_secotr_dict.get(market)
    if _sector == None:
        raise KeyError(f"不存在该市场:{market}")
    _now_sector = _sector[2:]
    
    
    # 过期上证和过期深证有专门的板块，不需要处理
    if market == "SHO" or market == "SZO":
        if data_type == 0:
            _list = xtdata.get_stock_list_in_sector(_now_sector)
        elif data_type == 1:
            _list = xtdata.get_stock_list_in_sector(_sector)
        elif data_type == 2:
            _list = xtdata.get_stock_list_in_sector(_sector) + xtdata.get_stock_list_in_sector(_now_sector)
        else:
            raise KeyError(f"data_type参数错误:{data_type}")
        return _list
        
    # 期货期权需要额外处理
    if data_type == 0:
        all_list = xtdata.get_stock_list_in_sector(_now_sector)
    elif data_type == 1:
        all_list = xtdata.get_stock_list_in_sector(_sector)
    elif data_type == 2:
        all_list = xtdata.get_stock_list_in_sector(_sector) + xtdata.get_stock_list_in_sector(_now_sector)
    else:
        raise KeyError(f"data_type参数错误:{data_type}")
    
    _list = []
    pattern1 = r'^[A-Z]{2}\d{4}-[A-Z]-\d{4}\.[A-Z]+$'
    pattern2 = r'^[a-zA-Z]+\d+[a-zA-Z]\d+\.[A-Z]+$'
    pattern3 = r'^[a-zA-Z]+\d+-[a-zA-Z]-\d+\.[A-Z]+$'
    for i in all_list:
        if re.match(pattern1,i):
            _list.append(i)
        elif re.match(pattern2,i):
            _list.append(i)
        elif re.match(pattern3,i):
            _list.append(i)
    # _list =[i for i in all_list if re.match(pattern, i)]
    return _list


def get_option_underline_code(code:str) -> str:
    """
    注意：该函数不适用与股指期货期权
    Todo: 根据商品期权代码获取对应的具体商品期货合约
    Args:
        code:str 期权代码
    Return:
        对应的期货合约代码
    """
    Exchange_dict = {
        "SHFE":"SF",
        "CZCE":"ZF",
        "DCE":"DF",
        "INE":"INE",
        "GFEX":"GF"
    }
    
    if code.split(".")[-1] not in [v for k,v in Exchange_dict.items()]:
        raise KeyError("此函数不支持该交易所合约")
    info = xtdata.get_option_detail_data(code)
    underline_code = info["OptUndlCode"] + "." + Exchange_dict[info["OptUndlMarket"]]

    return underline_code
        
        
def get_financial_futures_code_from_index(index_code:str) -> list:
    """
    ToDo:传入指数代码，返回对应的期货合约（当前）
    Args:
        index_code:指数代码，如"000300.SH","000905.SH"
    Retuen:
        list: 对应期货合约列表
    """
    financial_futures = xtdata.get_stock_list_in_sector("中金所")
    future_list = []
    pattern = r'^[a-zA-Z]{1,2}\d{3,4}\.[A-Z]{2}$'
    # s_dict = {re.findall(r"[a-zA-Z]+",i.split(".")[0])[0] : [] for i in future_list}
    for i in financial_futures:
        
        if re.match(pattern,i):
            future_list.append(i)
    ls = []
    for i in future_list:
        _info = xtdata._get_instrument_detail(i)
        _index_code = _info["ExtendInfo"]['OptUndlCode'] + "." + _info["ExtendInfo"]['OptUndlMarket']
        if _index_code == index_code:
            ls.append(i)
    return ls

# url='https://open.feishu.cn/open-apis/bot/v2/hook/11385e95-ec73-4bcb-94c8-3adfdc0fd94f'
def send_feishu_msg(url,msg):
    """
    Todo: 给飞书机器人发消息
    Arge:
        url:机器人的url
        msg:信息内容
    """
    headers = {
    'Content-Type': 'application/json'
    }
    data = {"msg_type": "text","content": {"text":msg}
    }
    post_data = json.dumps(data)
    res = requests.post(url=url, data=post_data, headers=headers)

