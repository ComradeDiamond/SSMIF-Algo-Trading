import importlib
import pandas
part1 = importlib.import_module("algo_part1_justin_chen")

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
        
        # The stock we want to invest in
        part1.IndicatorsTA.__init__(self, "SPY", "2023-01-01", "2023-03-05")

        # Initializes instance variables we might want down the line
        self.__quantity__ = quantity
        self.__position__ = position

        # Creates a cumulative dataframe that contains the date, and % returns from buy and hold vs my strategy
        self.cumulative_df = pandas.DataFrame({"Buy & Hold": self.__calcBuyHold__(), "Your Strategy": self.__calculateReturns__()}, index=self.stockdf.index)
    
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

        return diffData

    '''
    Contains and execute my algo strategy's logic
    Then, it stores and returns a DataFrame containing my % returns
    @returns {DataFrame} A dataframe containing my % returns
    '''
    def calculateReturns(self):
        ''

    '''
    This plots the returns of the "buy & hold" strategy vs my strategy
    It should plot percent returns and the profits of both strategies side to side.
    '''
    def PlotReturns(self):
        ''


sample = backtest(100, True)
print(sample.cumulative_df)