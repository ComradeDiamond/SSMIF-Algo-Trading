import pandas
import yfinance
from datetime import date, timedelta
import matplotlib.pyplot as plt
import mplcursors as mpc


'''
Helper function that returns the date $window days before $date
@returns {Date} The date $window days before $date 
@param {Date} date The Date to calculate from
@param {int} window The number of days before the startDate
'''
def daysBefore(date, window):
    return date - timedelta(days = window)

'''
Helper function that $window days after $date
@returns {Date} The date $window days after $date 
@param {Date} date The Date to calculate from
@param {int} window The number of days after the startDate
'''
def daysAfter(date, window):
    return date + timedelta(days = window)

'''
Helper function that finds the date difference
Date 2 - Date 1 basically
@returns {int} Date difference in days
'''
def dateDifference(date1, date2):
    return (date2 - date1).days

'''
Calculates a basic linear regression based on the x and y given. Then, take the linear regression and apply it to a given x value.
@param {float[]} x X-values
@param {float[]} y Y-values
@param {float} evalX The X to evaluate for under the regression
@returns {float} A y-value representing the "regressed" number
I used this site for the equation: https://www.statisticshowto.com/probability-and-statistics/regression-analysis/find-a-linear-regression-equation/
'''
def linearRegression(x, y, evalX):
    sumX = sum(x)
    sumY = sum(y)
    sumX2 = sum([xi ** 2 for xi in x])

    sumXY = 0
    for i in range(len(x)):
        sumXY += x[i] * y[i]
    
    intercept = (sumY * sumX2 - sumX * sumXY) / (len(x) * sumX2 - (sumX ** 2))
    slope = (len(x) * sumXY - sumX * sumY) / (len(x) * sumX2 - (sumX ** 2))

    return (slope * evalX) + intercept

'''
Class that tracks Stock Data fetched from yfinance and plots stock indicators (RSI, BBands, Adj Close, VWAP)
'''
class IndicatorsTA:

    '''
    Constructor for an instance of indicators TA. 
    Also initializes a dataframes representing this stock's prices/data from startDate to endDate
    @param {String} tickerSymbol Letters representing the stock symbol (ie. SPY)
    @param {String} startDate Start date in YYYY-MM-DD format
    @param {String} endDate End date in YYYY-MM-DD format
    '''
    def __init__(self, tickerSymbol, startDate, endDate):

        # Creates field variables with the inputs in case we need it later
        self.__tickerSymbol = tickerSymbol
        self.__startDate = startDate
        self.__endDate = endDate

        # Create the stock dataframe

        categories = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Adj Close"     # Adjusted close
        ]

        self.stockdf = yfinance.download(tickerSymbol, startDate, endDate).reindex(columns=categories).round(2)

        self.algodf = self.__getIndicators()

    '''
    Calculates the Relative Strength Index of a stock over $window days
    Not using Panda dataframe libraries too much because I'm not sure if I'm allowed to have the libraries do the work for me
    @param {int} window The time period, in days, to calculate RSI by
    @returns {DataFrame} A dataframe representing the RSI of the stock from startDate to endDate
    '''
    def __getRSI(self, window : int = 14):
        # We want to get the stock data 3 extra days before the initial date as we need to find the gain/loss on the first day as well, and those dates might be on weekends
        # RSI might be a bit off from ones online since RSI uses prior data, and online sites accumulate data for the, say, past few hundred days
        initDate = self.stockdf.iloc[0].name.date()
        preStartData = yfinance.download(self.__tickerSymbol, daysBefore(initDate, window + 3), daysAfter(initDate, 1))
        rsiData = []

        # The first element in RSI = 100 - (100/ 1 + RS ), where RS = (average gain over window / average loss over window)
        gain = 0
        loss = 0
        
        for i in range(1, len(preStartData.index)):

            # Edge case: date out of bounds due to extra stock data (usually on weekends)
            if (preStartData.iloc[i].name.date() < daysBefore(initDate, window - 1)):
                continue

            priceDiff = preStartData.iloc[i]["Close"] - preStartData.iloc[i - 1]["Close"]

            if (priceDiff < 0):
                loss += abs(priceDiff)
            elif (priceDiff > 0):
                gain += priceDiff

        avgGain = gain / window
        avgLoss = loss / window
        
        if (avgLoss == 0):
            rsiData.append(100) # Edge case that uses infinity - if average loss (and hence loss because math) is 0, RSI == 100
        else:
            rsiData.append(100 - (100 / (1 + (avgGain / avgLoss))))

        # Reset vars to use later
        loss = 0
        gain = 0

        # The subsequent RSI equation is the same, but average gain/loss = [(previous average change) x (window - 1) + current change] / window.
        for i in range(1, len(self.stockdf.index)):
            
            priceDiff = (self.stockdf.iloc[i]["Close"] - self.stockdf.iloc[i - 1]["Close"])

            if (priceDiff < 0):
                loss = abs(priceDiff)
                gain = 0
            elif (priceDiff > 0):
                loss = 0
                gain = priceDiff

            avgGain = (avgGain * (window - 1) + gain) / window
            avgLoss = (avgLoss * (window - 1) + loss) / window

            if (avgLoss == 0):
                rsiData.append(100) # Edge case that uses infinity - if average loss (and hence loss because math) is 0, RSI == 100
            else:
                rs = avgGain / avgLoss
                rsi = 100 - (100 / (1 + (rs)))
                rsiData.append(rsi)
        
        rsiData = map(lambda x: round(x, 2), rsiData)
        return pandas.DataFrame({"RSI(" + str(window) + ")" : rsiData}, index=self.stockdf.index)
    
    '''
    Calculates the left, right, and middle Bollinger Bands over the course of $period days returns a dataframe with them.
    @param {int} period The time period of the B Band
    @param {int} stdev Number of standard deviations for the middle and lower bands
    @returns {DataFrame} A dataframe with bband data
    '''
    def __getBBands(self, period : int = 20, stdev : int = 2):

        # Combine the items 20 days before with the stock prices now
        initDate = self.stockdf.iloc[0].name.date()

        preData = yfinance.download(self.__tickerSymbol, daysBefore(initDate, period), initDate).reindex(columns=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Adj Close"
        ])
    
        data = pandas.concat([preData, self.stockdf])
        
        # Start calculating bands. Initialize a queue and currentSum for the middle band
        # Utilizes a date range to fetch stock data. If the data for date doesn't exist (ie. it's a weekend), linear regression it

        queue = []
        currSum = 0

        bandDataFrame = {
            "LBand(" + str(period) + ", " + str(stdev) + ")": [],
            "MBand(" + str(period) + ", " + str(stdev) + ")": [],
            "HBand(" + str(period) + ", " + str(stdev) + ")": [],
        }

        for date in pandas.date_range(daysBefore(initDate, period), self.__endDate):
            timestamp = pandas.Timestamp.fromisoformat(date.isoformat())

            # If price doesn't exist, linear regression it
            if (not (timestamp in data.index)):
                if queue == []:
                    continue

                linRegNum = min(5, len(queue))      # Thought was we don't want too many elements in the linear regression because there might be, say, an upward trend that just happened 1-2 days ago
                linRegResult = linearRegression(range(1, linRegNum + 1), queue[len(queue) - linRegNum : len(queue)], linRegNum + 1)
                queue.append(linRegResult)
                currSum += linRegResult
            else:
                # Add stuff to queue and currSum so we can math
                queue.append(data.loc[timestamp]["Close"])
                currSum += data.loc[timestamp]["Close"]
            
            # Remove items from queue to preserve 20 day average if needed
            if (len(queue) > 20):
                currSum -= queue.pop(0)

            if (date >= initDate and (timestamp in data.index)):
                # The middle band is a 20 day simple moving average
                middleBand = currSum / len(queue)

                # The lower and middle band requires standard deviation. So let's find that. Fortunately, the middle band is the mean
                std = (sum((xi - middleBand) ** 2 for xi in queue) / len(queue)) ** (0.5)

                lowerBand = middleBand - stdev * std
                upperBand = middleBand + stdev * std
                
                bandDataFrame["LBand(" + str(period) + ", " + str(stdev) + ")"].append(round(lowerBand, 2))
                bandDataFrame["MBand(" + str(period) + ", " + str(stdev) + ")"].append(round(middleBand, 2))
                bandDataFrame["HBand(" + str(period) + ", " + str(stdev) + ")"].append(round(upperBand, 2))

        return pandas.DataFrame(bandDataFrame, index=self.stockdf.index)

    '''
    Aggregates the daily VWAP value in a DataFrame. 
    Since VWAP consists of a collection of Volume * Price lf Volume is Traded at throughout the day, we download additional ytf stock data for each 15 minute interval of the day
    @returns {DataFrame} The Volume Weighted Average Price of the stock from startDate to endDate
    '''
    def __getVWAP(self):
        
        # This list will contain all the aggregated vwaps
        vwaps = []

        # Calculate the VWAP for every day within self.stockdf
        for k in range(len(self.stockdf.index)):
            dateStr = self.stockdf.iloc[k].name.date()
            daysFromToday = dateDifference(dateStr, date.today())
            currentSeries = self.stockdf.iloc[k]

            # Since VWAPs require the aggregate of Volume * Price of Volume throughout the day, and since yfinance has limitations on how much
            # precise data it stores going back, we'll try to maximize the number of intervals we can aggregate using VWAP.
            if (daysFromToday <= 725):

                dailyData = yfinance.download(self.__tickerSymbol, dateStr.isoformat(), daysAfter(dateStr, 1), interval="1h")
        
                cumPV = 0
                # VWAP formula: (cumulative typical price * volume at interval) / total volume for the day
                for i in range(len(dailyData.index)):
                    typicalPrice = (dailyData.iloc[i]["Close"] + dailyData.iloc[i]["High"] + dailyData.iloc[i]["Low"]) / 3
                    cumPV += typicalPrice * dailyData.iloc[i]["Volume"]
                
                vwap = cumPV / currentSeries["Volume"]
                vwaps.append(round(vwap, 2))
            else:
                # VWAP formula: (cumulative typical price * volume at interval) / total volume for the day
                # But since we can only access 1d intervals, it simplifies to cumulative price
                typicalPrice = (currentSeries["High"] + currentSeries["Close"] + currentSeries["Low"]) / 3
                vwaps.append(round(typicalPrice, 2))

        return pandas.DataFrame({"VWAP": vwaps}, index=self.stockdf.index)

    '''
    @returns {DataFrame} Containing the Adjusted Close Price, RSI, LBand, MBand, HBand, VWAP of the stock from startDate to endDate
    '''
    def __getIndicators(self):
        rsi = self.__getRSI()
        bbands = self.__getBBands()
        vwap = self.__getVWAP()
        adjClose = self.stockdf[["Adj Close"]]
        
        merged = pandas.merge(adjClose, rsi, left_index=True, right_index=True).merge(bbands, left_index=True, right_index=True).merge(vwap, left_index=True, right_index=True)

        # print(merged)
        return merged

    '''
    Plots self.algodf in another window containing Adj Close, RSI, BBands, and VWAP data
    This uses pyplot.
    '''
    def plotIndicators(self):
        figure, graphs = plt.subplots(3, sharex=True)
        (graphRSI, graphBands, graphVWAP) = graphs

        figure.suptitle("Algodf Results for $" + self.__tickerSymbol, fontsize=32, fontname="Trebuchet MS")
        figure.text(0.4, 0.05, "Hover Over a Line to Show Data", fontsize=14, fontname="Trebuchet MS")
        figure.set_size_inches(12, 6)

        # Allocate the lists to graph first
        times = list(map(lambda x: x.date().isoformat() , self.algodf.index))
        adjClose = self.algodf.iloc[:, 0].values
        rsi = self.algodf.iloc[:, 1].values
        lowBand = self.algodf.iloc[:, 2].values
        midBand = self.algodf.iloc[:, 3].values
        upperBand = self.algodf.iloc[:, 4].values
        vwap = self.algodf.iloc[:, 5].values
        
        # Giving these all variable names so they can be strong references for figure.legend and mpccursor
        graphRSI.plot(times, rsi, color="#30e8fc", label="RSI")
        graphVWAP.plot(times, vwap, color="#fc6afc", label="VWAP")
        graphBands.plot(times, lowBand, color="#ff6145", label="Low Band")
        graphBands.plot(times, midBand, color="#fcde30", label="Middle Band")
        graphBands.plot(times, upperBand, color="#30fc33", label="Upper Band")
        graphBands.plot(times, adjClose, color="#1603a3", label="Adjusted Close")

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
            lastAxvLine[0] = graph.axes.axvline(x=selected.target[0], ymin=0, ymax=3.3, c="#1551d4", clip_on=False, alpha=0.5, dashes=(5,1,5,2), lw=2)

            # Gets date from the initial annotation
            currText = selected.annotation.get_text()
            currDate = currText[currText.find("x=") + 2 : currText.find("y=")].strip()
            indexDate = times.index(currDate)

            selected.annotation.set_bbox(dict(boxstyle="round", alpha=0.7, color='#ededed'))
            return selected.annotation.set_text(f"{currDate}\nRSI: {rsi[indexDate]}\nAdj Close: ${adjClose[indexDate]}\n" +
            f"Lower BBand: ${lowBand[indexDate]}\nMiddle BBand: ${midBand[indexDate]}\nUpper BBand: ${upperBand[indexDate]}\nVWAP: ${vwap[indexDate]}")
        
        lastAxvLine = [None]
        mpc.cursor(graphs, hover=True).connect(
            "add", lambda x: cursorStuff(x, lastAxvLine))
    
        plt.show()


# test = IndicatorsTA("SPY", "2023-01-01", "2023-03-05")
# test.plotIndicators()