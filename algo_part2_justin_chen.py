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
A class that extends IndicatorsTA
Backtests the "buy and hold" strategy against my other trading strategy
'''
class backtest(part1.IndicatorsTA):
    '''
    Initializes the backtest class object
    @param {float} quantity The Number of Shares you're buying in the buy and hold strategy
    @param {boolean} position True if you want a long position, False if you want a short position
    '''
    def __init__(self, quantity, position):
        
        # The stock we want to invest in. Change it as we like
        part1.IndicatorsTA.__init__(self, "AAPL", "2023-01-01", "2023-03-05")

        # Initializes instance variables we might want down the line
        self.__quantity__ = quantity
        self.__position__ = position

        # Useful dataframe
        self.trading_df = self.__generateTradingDF()

        # Creates a cumulative dataframe that contains the date, and % returns from buy and hold vs my strategy
        self.cumulative_df = pandas.DataFrame({"Buy & Hold": self.__calcBuyHold__(), "Your Strategy": self.calculateReturns()}, index=self.stockdf.index)
    
    '''
    Helper Function that calculates % returns if you buy and hold.
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

        return list(map(lambda x: round(x, 3), diffData))

    '''
    A private helper function that generates a dataframe containing the important stock indicators I will use for my trading strategy
    @returns {DataFrame} A dataframe with start-end dates as the index, OBV, 1st and 2nd derivative of the OBV, BBands, and MACD
    '''
    def __generateTradingDF(self):
        # Get finance data 25 days (since EMA26 - will explain later - will be calculated 
        # by 25 previous days + 1) before and merge it with self.stockdf
        startDate = self.stockdf.iloc[0].name.date()
        preStartData = yfinance.download(self.tickerSymbol, part1.daysBefore(startDate, 25), part1.daysAfter(startDate, 1)).reindex(columns=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Adj Close"     # Adjusted close
        ]).round(2)

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
            "MACD": []
        }

        # To find MACD, you need to find EMA(26) and EMA(12)
        # Smoothing EMA set to 2, formula for smoothing multiplier (smoothing)/(1 + days)
        emaSmoothingMul26 = 2/27
        emaSmoothingMul12 = 2/13

        # Iterates through all days to calculate OBV and MACD
        for i in range(1, len(mergedDf.index)):
            currData = mergedDf.iloc[i]

            # First data entries
            if (i == 0):
                fullObv.append(0)       # OBV is 0 for the first data entry since previous day close doesn't exist
                
                # Since EMA(previous day) doesn't exist, both EMAs will just me closing price * multiplier
                fullEMA12.append(emaSmoothingMul12 * currData["Close"])
                fullEMA26.append(emaSmoothingMul26 * currData["Close"])

            # Not first data entry
            else:
                lastColData = mergedDf.iloc[i- 1]

                # 1) First calculate the EMA of 12 and 16
                # EMA Formula = Closing price x multiplier + EMA (previous day) x (1-multiplier)
                fullEMA12.append(currData["Close"] * emaSmoothingMul12 + fullEMA12[i - 1] * (1 - emaSmoothingMul12))
                fullEMA26.append(currData["Close"] * emaSmoothingMul26 + fullEMA26[i - 1] * (1 - emaSmoothingMul26))

                # 2) Calculate OBV. It's previous OBV + volume is close today > close yesterday, 
                # negative ifvice versa, and 0 if the close is the same
                currObv = fullObv[i - 1]

                if (currData["Close"] > lastColData["Close"]):
                    currObv += currData["Volume"]
                elif (currData["Close"] < lastColData["Close"]):
                    currObv -= currData["Volume"]
        
        # Calculate the MACD. It's just EMA12 - EMA26. 
        # Shave off the first 25 data entries though because that was only used to make MACD/OBV more accurate
        data["MACD"] = [fullEMA12[k] - fullEMA26[k] for k in range(26, len(fullEMA12))]

        # Get the OBV we'll actually use
        data["obv"] = fullObv[26:]

        # Uses basic linear regression to approximate the rate of change of the OBV in ~~this economy~~ the timeframe
        # Just let initial ROC equal 0 because the slope might as well be flat.
        # At the same time for optimization, calculate how fast that ROC is changing.
        fullObvROC = [0]
        fullObvROC2 = [0]

        for i in range(1, len(fullObv)):

            # The number to go back should ideally be around 10 (based it off 14 day SMA - the weekends)
            # If there's not 10 days of data, just use what we can.
            numberToGoBack = min(10, i)
            firstDerivative = getRateOfChange(fullObv[i - numberToGoBack : i + 1], range(i + 1))
            fullObvROC.append(firstDerivative)

            # At the same time find second "derivative"
            secondDerivative = getRateOfChange(fullObvROC[i - numberToGoBack : i + 1], range(i + 1))
            fullObvROC2.append(secondDerivative)

        data["dOBVdX"] = fullObvROC[26:]
        data["d2OBVdX2"] = fullObvROC2[26:]

        return pandas.DataFrame(data, index=self.stockdf.index)
    
    '''
    Contains and execute my algo strategy's logic
    Then, it stores and returns a DataFrame containing my % returns
    @returns {float[]} An array containing my % returns
    '''
    def calculateReturns(self):
        # OBV
        # BBands - https://www.schwab.com/learn/story/bollinger-bands-what-they-are-and-how-to-use-them#:~:text=A%20Bollinger%20Band%20consists%20of,of%20price%2C%20thus%20incorporating%20volatility.
        # MACD
        ''

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
        percentMyStrat = self.cumulative_df.iloc[:, 0].values

        startStockPrice = self.stockdf.iloc[0]["Close"]

        moneyBuyHold = list(map(lambda x : round(x * startStockPrice * self.__quantity__, 2), percentBuyHold))
        moneyMyStrat = list(map(lambda x : round(x * startStockPrice * self.__quantity__, 2), percentMyStrat))

        graphPercent.plot(times, percentBuyHold, color="#ffde0a", label="Buy & Hold")
        graphPercent.plot(times, percentMyStrat, color="#43ff0a", label="My Strat")
        graphMoney.plot(times, moneyBuyHold, color="#ffde0a")
        graphMoney.plot(times, moneyMyStrat, color="#43ff0a")

        graphPercent.set_title("Percentage %", fontsize=18, fontname="Trebuchet MS", pad=10)
        graphMoney.set_title("Profit", fontsize=18, fontname="Trebuchet MS", pad=10)

        # Constrain % y
        graphPercent.set_ylim(-1, 1)

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
            return selected.annotation.set_text("Hi")
        
        lastAxvLine = [None]
        mpc.cursor(graphs, hover=True).connect(
            "add", lambda x: cursorStuff(x, lastAxvLine))
    
        plt.show()

sample = backtest(100, True)
# sample.PlotReturns()