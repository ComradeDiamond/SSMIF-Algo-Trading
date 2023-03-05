import pandas
import yfinance

'''
Class that tracks Stock Indicators, adds a display to said indicators, and converts the String format to a usable JSON-like structure
'''
class IndicatorsTA:

    '''
    Constructor for an instance of indicators TA. 
    Also initializes a dataframes for this particular stock indicator as instance vars.
    @param {String} tickerSymbol Letters representing the stock symbol (ie. SPY)
    @param {String} startDate Start date in YYYY-MM-DD format
    @param {String} endDate End date in YYYY-MM-DD format
    '''
    def __init__(self, tickerSymbol, startDate, endDate):
        # Initializing field vars just in case we need them down the line
        self.tickerSymbol = tickerSymbol
        self.startDate = startDate
        self.endDate = endDate

        # Create the stock dataframe

        categories = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Adj Close"     # Adjusted close
        ]

        self.stockdf = yfinance.download(tickerSymbol, startDate, endDate)
        self.stockdf = self.stockdf.reindex(columns=categories)

        # Call algo df
        self.algodf = self.getIndicators()
    
    def getIndicators(self):
        "implement"

test = IndicatorsTA("AAPL", "2017-01-01", "2017-04-30")
print(test.stockdf)