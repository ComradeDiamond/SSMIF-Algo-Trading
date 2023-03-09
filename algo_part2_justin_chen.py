import importlib
import pandas
import matplotlib.pyplot as plt
import mplcursors as mpc
import yfinance
from datetime import date, timedelta
part1 = importlib.import_module("algo_part1_justin_chen")

'''
Calculates a basic linear regression based on the x and y given. However, only return the slope.
@param {float[]} x X-values
@param {float[]} y Y-values
@returns {float} The rate of change.
'''
def getRateOfChange(x, y):
    sumX = sum(x)
    sumY = sum(y)
    sumX2 = sum([xi ** 2 for xi in x])

    sumXY = 0
    for i in range(len(x)):
        sumXY += x[i] * y[i]
    
    slope = (len(x) * sumXY - sumX * sumY) / (len(x) * sumX2 - (sumX ** 2))

    return slope

'''
Helper function that finds the date difference
Date 2 - Date 1 basically
@returns {int} Date difference in days
'''
def dateDifference(date1, date2):
    return (date2 - date1).days

'''
A class that extends IndicatorsTA
Backtests the "buy and hold" strategy against my other trading strategy
This initializes to trading the S&P 500 during the past year
'''
class backtest(part1.IndicatorsTA):
    '''
    Initializes the backtest class object
    @param {float} quantity The Number of Shares you're buying in the buy and hold strategy
    @param {boolean} position True if you want a long position, False if you want a short position
    '''
    def __init__(self, quantity, position):
        
        # The stock we want to invest in. Change it as we like
        part1.IndicatorsTA.__init__(self, "SPY", "2022-03-01", "2023-02-28")

        # Initializes instance variables we might want down the line
        self.__quantity__ = quantity
        self.__position__ = position

        # Useful dataframe
        self.trading_df = self.__generateTradingDF()

        # Creates a cumulative dataframe that contains the date, and % returns from buy and hold vs my strategy
        self.cumulative_df = pandas.DataFrame({"Buy & Hold": self.__calcBuyHold__(), "Your Strategy": self.calculateReturns()}, index=self.stockdf.index)
    
    '''
    Helper Function that calculates % returns if you buy and hold.
    We don't calculate how many stocks we hold because they cancel out in the percent change equation
    Takes into account if you're in a long or short position
    @returns {float[]} An array of floats that represent % returns for each day from StartDate to EndDate
    '''
    def __calcBuyHold__(self):
        # Formula: New - Original / Original

        # Creates an array to track the diff data
        diffData = []
        initPrice = None

        for currPrice in self.stockdf.iloc[:, 3].values:
            if (initPrice == None):         # This means we're at first element
                diffData.append(0)
                initPrice = currPrice
            else:
                diffData.append((currPrice - initPrice) / initPrice)

        priceArr = list(map(lambda x: round(x, 3), diffData))

        # If you are in a long position, rise in price == profit
        if (self.__position__):
            return priceArr
        else: # The reverse is true for shorts
            return list(map(lambda x: x * -1), diffData)

    '''
    A private helper function that generates a dataframe containing the important stock indicators I will use for my trading strategy
    @returns {DataFrame} A dataframe with start-end dates as the index, OBV, 1st and 2nd derivative of the OBV, BBands, and MACD
    '''
    def __generateTradingDF(self):
        # Get finance data 25 days (since EMA26 - will explain later - will be calculated 
        # by 25 previous days + 1) before and merge it with self.stockdf
        startDate = self.stockdf.iloc[0].name.date()
        preStartData = yfinance.download(self.tickerSymbol, part1.daysBefore(startDate, 25), startDate).reindex(columns=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Adj Close"     # Adjusted close
        ]).round(2)
        preStartDataLen = len(preStartData.index)

        mergedDf = pandas.concat([preStartData, self.stockdf])

        # Since we're going to be taking the 1st and 2nd degree rate of changes of the OBV and the MACD over a time span, 
        # we'll need this array to store everything in first.

        # First item of an OBV is 0, and first item of an EMA 
        fullObv = []
        fullEMA12 = []
        fullEMA26 = []

        # The data goes here.
        data = {
            "obv": [],
            "dOBVdX": [],
            "d2OBVdX2": [],
            "lowerBand": self.algodf.iloc[:, 2].values,
            "middleBand": self.algodf.iloc[:, 3].values,
            "upperBand": self.algodf.iloc[:, 4].values,
            "MACD": [],
            "dMACDdX": [],
        }

        # To find MACD, you need to find EMA(26) and EMA(12)
        # Smoothing EMA set to 2, formula for smoothing multiplier (smoothing)/(1 + days)
        emaSmoothingMul26 = 2/27
        emaSmoothingMul12 = 2/13

        # Iterates through all days to calculate EMA
        for k in range(len(mergedDf.index)):
            dateStr = mergedDf.iloc[k].name.date()

            if (dateDifference(startDate, dateStr) >= 0):
                currData = mergedDf.iloc[k]
                # If this is the first time we find EMA12, take its SMA12
                if (len(fullEMA12) == 0):
                    acc = 0
                    numAdded = 0
                    for i in range(k + 1):
                        if (dateDifference(mergedDf.iloc[i].name.date(), startDate) < 12):
                            acc += mergedDf.iloc[i]["Close"]
                            numAdded += 1
                    
                    sma = (acc / numAdded)
                    fullEMA12.append(sma)
                else:
                    # EMA Formula = Closing price x multiplier + EMA (previous day) x (1-multiplier)
                    fullEMA12.append(currData["Close"] * emaSmoothingMul12 + fullEMA12[len(fullEMA12) - 1] * (1 - emaSmoothingMul12))

                # If this is the first time we find EMA26, take its SMA26
                if (len(fullEMA26) == 0):
                    acc = 0
                    for i in range(k + 1):
                        acc += mergedDf.iloc[i]["Close"]
                
                    sma = (acc / (k + 1))
                    fullEMA26.append(sma)
                else:
                    fullEMA26.append(currData["Close"] * emaSmoothingMul26 + fullEMA26[len(fullEMA26) - 1] * (1 - emaSmoothingMul26))
        
        # Calculate the MACD. It's just EMA12 - EMA26. 
        data["MACD"] = [fullEMA12[k] - fullEMA26[k] for k in range(len(fullEMA12))]

        # Calculate the almost instantaneous rate of change of the MACD
        for k in range(len(data["MACD"])):

            # Go back 3 spaces, including the current item
            numberToGoBack = min(3 - 1, k)
            firstDerivative = getRateOfChange(range(numberToGoBack + 1), data["MACD"][k - numberToGoBack : k + 1])
            data["dMACDdX"].append(firstDerivative)

        # Iterates through all days to calculate OBV
        for i in range(len(mergedDf.index)):
            currData = mergedDf.iloc[i]

            # First data entries
            if (i == 0):
                fullObv.append(0)       # OBV is 0 for the first data entry since previous day close doesn't exist
            else: # Not first data entry
                lastColData = mergedDf.iloc[i- 1]

                # Calculate OBV. It's previous OBV + volume is close today > close yesterday, 
                # negative ifvice versa, and 0 if the close is the same
                currObv = fullObv[i - 1]

                if (currData["Close"] > lastColData["Close"]):
                    currObv += currData["Volume"]
                elif (currData["Close"] < lastColData["Close"]):
                    currObv -= currData["Volume"]
                
                fullObv.append(currObv)

        # Get the OBV we'll actually use
        data["obv"] = fullObv[preStartDataLen:]

        # Uses basic linear regression to approximate the rate of change of the OBV in ~~this economy~~ the timeframe
        # Just let initial ROC equal 0 because the slope might as well be flat.
        # At the same time for optimization, calculate how fast that ROC is changing.
        fullObvROC = [0]
        fullObvROC2 = [0]

        for i in range(1, len(fullObv)):

            # The number to go back should ideally be around 10 (based it off 14 day SMA - the weekends)
            # If there's not 10 days of data, just use what we can.
            numberToGoBack = min(10 - 1, i)
            firstDerivative = getRateOfChange(range(numberToGoBack + 1), fullObv[i - numberToGoBack : i + 1])
            fullObvROC.append(firstDerivative)

            # At the same time find second "derivative"
            secondDerivative = getRateOfChange(range(numberToGoBack + 1), fullObvROC[i - numberToGoBack : i + 1])
            fullObvROC2.append(secondDerivative)

        data["dOBVdX"] = fullObvROC[preStartDataLen:]
        data["d2OBVdX2"] = fullObvROC2[preStartDataLen:]

        return pandas.DataFrame(data, index=self.stockdf.index)
    
    '''
    Contains and execute my algo strategy's logic
    Then, it stores and returns a DataFrame containing my % returns
    @returns {float[]} An array containing my % returns
    '''
    def calculateReturns(self) -> pandas.Dataframe:
        # OBV
        # BBands - https://www.schwab.com/learn/story/bollinger-bands-what-they-are-and-how-to-use-them#:~:text=A%20Bollinger%20Band%20consists%20of,of%20price%2C%20thus%20incorporating%20volatility.
        # MACD

        # This strategy is similar to trend-following. We use OBV and its 1st and second "derivatives" as the
        # most important baseline to determine *if* we should buy or sell since the direction of call volumes is
        # indicative of a trend. Then, we use MACD and its 1st "derivative" (since MACD already measures change) to determine *when* we should 
        # sell to maximize profit. We then use bbands (more specifically, the difference between the upper and lower)
        # BBand to determine how much we should buy or sell since if a stock is volatile, we want to get rid of it
        # ASAP

        # If long
        if (self.__position__):
            # Establish some initial variables
            bank = 0                                # when we sell/buy, money comes from here
            returns = [0]                            # track % returns
            initMoney = self.stockdf.iloc[0]["Close"] * self.__quantity__
            stocksWeHave = self.__quantity__        # start with self.quantity number of stocks

            # Adding this as internal cooldown to trades
            justSold = 0
            justBought = 0
            
            # Looping through the dates in self.trading_df simulates every new day while trading
            for i in range(1, len(self.trading_df.index)):
                currDate = self.trading_df.iloc[i].name.date().isoformat()
                currData = self.trading_df.iloc[i]
                currPrice = self.stockdf.iloc[i]["Close"]
                
                # Business' booming, we should sell our longs
                if (currData["dOBVdX"] > 0 and currData["d2OBVdX2"] > 0):
                    if (currData["MACD"] > 0 and currData["dMACDdX"] < 0.3 and justSold == 0):      # Critical point, or about to be critical point
                        # print(currDate + "sell")
                        # print(bank)
                        justSold = 5
                        # Calculate the difference between the bands and compare it to the middle band (SMA)
                        # By basing it off SPY, < 3% is less voltaility     - Sell 25%
                        # 3 - 7% is medium                                  - Sell 50%
                        # > 7 is pretty big volatility ngl                  - Sell 75% 
                        # > 10                                              - Sell 90%

                        percentageFromSMA = (currData["upperBand"] - currData["lowerBand"]) / currData["middleBand"]

                        if (percentageFromSMA < 0.03):
                            stockSell = round(stocksWeHave * 0.25)
                            bank += currPrice * (stockSell)
                            stocksWeHave = stocksWeHave - stockSell
                        elif (percentageFromSMA < 0.07):
                            stockSell = round(stocksWeHave * 0.5)
                            bank += currPrice * (stockSell)
                            stocksWeHave = stocksWeHave - stockSell
                        elif (percentageFromSMA < 0.1):
                            stockSell = round(stocksWeHave * 0.75)
                            bank += currPrice * (stockSell)
                            stocksWeHave = stocksWeHave - stockSell
                        else:
                            stockSell = round(stocksWeHave * 0.9)
                            bank += currPrice * (stockSell)
                            stocksWeHave = stocksWeHave - stockSell
                elif (currData["dOBVdX"] < 0 and currData["d2OBVdX2"] < 0): # this is not cash money
                    if (currData["MACD"] < 0 and currData["dMACDdX"] > -0.3 and currData["MACD"] < -4 and justBought == 0):        # Critical point, it rising? o.o
                        # Volatility, but this time, you buy as a % of existing stocks.
                        # Buy 25, 50, 75% of the money in our bank
        
                        if (bank > currPrice):
                            # print(currDate + "buy")
                            # print(bank)
                            justBought = 5
                            # You need to be able to afford 1 stock to do anything
                            percentageFromSMA = (currData["upperBand"] - currData["lowerBand"]) / currData["middleBand"]

                            if (percentageFromSMA < 0.03):
                                stockBuy = round(bank * 0.25 / currPrice)
                                bank -= currPrice * stockBuy
                                stocksWeHave = stocksWeHave + stockBuy
                            elif (percentageFromSMA < 0.07):
                                stockBuy = round(bank * 0.5 / currPrice)
                                bank -= currPrice * (stockBuy)
                                stocksWeHave = stocksWeHave + stockBuy
                            else:
                                stockBuy = round(bank * 0.75 / currPrice)
                                bank -= currPrice * (stockBuy)
                                stocksWeHave = stocksWeHave + stockBuy

                # Calculate our net returns by end of day
                currMoney = bank + stocksWeHave * currPrice
                returns.append((currMoney - initMoney)/initMoney)
                justSold = max(0, justSold - 1)
                justBought = max(0, justBought - 1)

            return returns
            
        else:
            # Shorts. We're kinda already at a disadvantage because we hold falling stocks.
            # Rewriting this whole section even though the math is similar to the one for long
            # since so many variables and tiny intricacies are different here

            returns = [0]                            # track % returns
            initMoney = self.stockdf.iloc[0]["Close"] * self.__quantity__
            bank = initMoney                         # In a shorts scenario, we start with initMoney in the band. It's just that in buy and hold... we do nothing and watch prices fall
            stocksWeOwe = 0                          # start with self.quantity number of stocks
            justSold = 0
            justBought = 0

            # Looping through the dates in self.trading_df simulates every new day while trading
            for i in range(1, len(self.trading_df.index)):
                currData = self.trading_df.iloc[i]
                currPrice = self.stockdf.iloc[i]["Close"]

                # Business' booming, Short this thing.                             
                if (currData["dOBVdX"] > 0 and currData["d2OBVdX2"] > 0):
                    if (currData["MACD"] > 0 and currData["dMACDdX"] < 0.3 and justSold == 0):      # Critical point, or about to be critical point
                        # Calculate the difference between the bands and compare it to the middle band (SMA)
                        # By basing it off SPY, < 3% is less voltaility     - Short 20% of our bank
                        # 3 - 7% is medium                                  - Short 40% of our bank
                        # > 7 is pretty big volatility                      - Short 60% of our bank
                        justSold = 5
                        percentageFromSMA = (currData["upperBand"] - currData["lowerBand"]) / currData["middleBand"]

                        if (percentageFromSMA < 0.03):
                            makeShortPromise = round(bank * 0.20 / currPrice)
                            bank += currPrice * (makeShortPromise)
                            stocksWeOwe += makeShortPromise
                        elif (percentageFromSMA < 0.07):
                            makeShortPromise = round(bank * 0.40 / currPrice)
                            bank += currPrice * (makeShortPromise)
                            stocksWeOwe += makeShortPromise
                        else:
                            makeShortPromise = round(bank * 0.60 / currPrice)
                            bank += currPrice * (makeShortPromise)
                            stocksWeOwe += makeShortPromise
                elif (currData["dOBVdX"] < 0 and currData["d2OBVdX2"] < 0): # this is not cash money. Buy some stonks to pay it back
                    if (currData["MACD"] < 0 and currData["dMACDdX"] > -0.3 and justBought == 0):        # Critical point, it rising? o.o
                        # Volatility, but this time, you buy as a % of existing stocks.
                        # Pay back 30%, 50%, 80% of the shorts we have

                        percentageFromSMA = (currData["upperBand"] - currData["lowerBand"]) / currData["middleBand"]

                        if (bank > currPrice):
                            justBought = 5
                            # You need to be able to afford at least 1 stock
                            if (percentageFromSMA < 0.03):
                                stockBuy = round(stocksWeOwe * 0.3)
                                bank -= currPrice * (stockBuy)
                                stocksWeOwe -= stockBuy
                            elif (percentageFromSMA < 0.07):
                                stockBuy = round(stocksWeOwe * 0.5)
                                bank -= currPrice * (stockBuy)
                                stocksWeOwe -= stockBuy
                            else:
                                stockBuy = round(stocksWeOwe * 0.8)
                                bank -= currPrice * (stockBuy)
                                stocksWeOwe -= stockBuy

                # Calculate our net returns by end of day
                # The stocks right now are a liability
                currMoney = bank - stocksWeOwe * currPrice
                returns.append((currMoney - initMoney)/initMoney)
                justBought = max(0, justBought - 1)
                justSold = max(0, justSold - 1)

            return returns

    '''
    This plots the returns of the "buy & hold" strategy vs my strategy
    It should plot percent returns and the profits of both strategies side to side.
    '''
    def PlotReturns(self):
        figure, graphs = plt.subplots(2, sharex=True, constrained_layout=True)
        (graphPercent, graphMoney) = graphs

        figure.suptitle("Trade Results", fontsize=24, fontname="Trebuchet MS")
        figure.set_size_inches(12, 6)

        # Allocate the lists to graph first
        times = list(map(lambda x: x.date().isoformat() , self.algodf.index))
        percentBuyHold = self.cumulative_df.iloc[:, 0].values
        percentMyStrat = self.cumulative_df.iloc[:, 1].values

        startStockPrice = self.stockdf.iloc[0]["Close"]

        # The money from strats has to be calculated based on the initial stock price
        moneyBuyHold = list(map(lambda x : round(x * startStockPrice * self.__quantity__, 2), percentBuyHold))
        moneyMyStrat = list(map(lambda x : round(x * startStockPrice * self.__quantity__, 2), percentMyStrat))

        graphPercent.plot(times, percentBuyHold, color="#ffde0a", label="Buy & Hold")
        graphPercent.plot(times, percentMyStrat, color="#43ff0a", label="My Strat")
        graphMoney.plot(times, moneyBuyHold, color="#ffde0a")
        graphMoney.plot(times, moneyMyStrat, color="#43ff0a")

        graphPercent.set_title("Percentage %", fontsize=18, fontname="Trebuchet MS", pad=10)
        graphMoney.set_title("Profit", fontsize=18, fontname="Trebuchet MS", pad=10)

        # Constrain % y
        # graphPercent.set_ylim(-1, 1)

        for graph in graphs:
            graph.axes.spines["top"].set_visible(False)
            graph.axes.spines["right"].set_visible(False)
            graph.yaxis.grid(True, color="#dedede")
            graph.xaxis.set_visible(False) 

        figure.legend(bbox_to_anchor=(1, 0.9), loc="upper right", borderaxespad=1, edgecolor="#f0f0f0")
        
        # Add a line describing all the data on the date
        def cursorStuff(selected, lastAxvLine):
            if (lastAxvLine[0] != None):
                lastAxvLine[0].remove()
            lastAxvLine[0] = graph.axes.axvline(x=selected.target[0], ymin=0, ymax=2.25, c="#080145", clip_on=False, alpha=0.7, dashes=(5,1,5,2), lw=2)

            # Gets date from the initial annotation
            currText = selected.annotation.get_text()
            currDate = currText[currText.find("x=") + 2 : currText.find("y=")].strip()
            indexDate = times.index(currDate)

            selected.annotation.set_bbox(dict(boxstyle="round", alpha=0.7, color='#ededed'))
            return selected.annotation.set_text(f"{currDate}\nBuy & Hold: {round(percentBuyHold[indexDate] * 100, 2)}%\nMy Strat: {round(percentMyStrat[indexDate] * 100, 2)}%\n"+
                f"Buy & Hold Profits: ${round(moneyBuyHold[indexDate], 2)}\nMy Profits: ${round(moneyMyStrat[indexDate], 2)}")
        
        lastAxvLine = [None]
        mpc.cursor(graphs, hover=True).connect(
            "add", lambda x: cursorStuff(x, lastAxvLine))
    
        plt.show()

# Start with 100 shares of S&P 500 and see the results
sample = backtest(100, True)
sample.PlotReturns()