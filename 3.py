import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download necessary resources for VADER sentiment analysis
nltk.download('vader_lexicon')


class StockTrendAnalysisTool:
    def __init__(self):
        # Page configuration
        st.set_page_config(
            page_title="AI Stock Trend Analysis Tool",
            page_icon="📈",
            layout="wide"
        )
        self.sia = SentimentIntensityAnalyzer()

    def calculate_rsi(self, data: pd.DataFrame, periods: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)"""
        delta = data['Close'].diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[
        pd.Series, pd.Series, pd.Series]:
        """Calculate Moving Average Convergence Divergence (MACD)"""
        exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def generate_buy_sell_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy and sell signals based on technical indicators"""
        # RSI Signals
        rsi = self.calculate_rsi(data)

        # MACD Signals
        macd, signal_line, histogram = self.calculate_macd(data)

        # Combine signals
        data['RSI'] = rsi
        data['MACD'] = macd
        data['Signal_Line'] = signal_line
        data['MACD_Histogram'] = histogram

        # Buy/Sell Logic
        data['Buy_Signal'] = (
                (rsi < 30) &  # Oversold condition
                (macd > signal_line)  # MACD crossing above signal line
        )

        data['Sell_Signal'] = (
                (rsi > 70) &  # Overbought condition
                (macd < signal_line)  # MACD crossing below signal line
        )

        return data

    def clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis"""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
        text = text.lower()
        return text

    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of the given text"""
        cleaned_text = self.clean_text(text)
        score = self.sia.polarity_scores(cleaned_text)['compound']

        if score > 0.05:
            sentiment = 'Positive'
        elif score < -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        return {'score': score, 'category': sentiment}

    def plot_stock_data(self, data: pd.DataFrame):
        """Create interactive stock price chart with buy/sell signals"""
        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ))

        # Buy signals
        buy_signals = data[data['Buy_Signal']]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['Low'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='green'),
            name='Buy Signal'
        ))

        # Sell signals
        sell_signals = data[data['Sell_Signal']]
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals['High'],
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color='red'),
            name='Sell Signal'
        ))

        fig.update_layout(
            title='Stock Price with Buy/Sell Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            height=600
        )

        return fig

    def run(self):
        """Main Streamlit application"""
        st.title("🚀 AI-Powered Stock Trend Analysis Tool")

        # Sidebar
        with st.sidebar:
            st.header("Stock Selection")
            stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL")
            timeframe = st.selectbox(
                "Select Timeframe",
                ["1mo", "3mo", "6mo", "1y", "5y"],
                index=2
            )
            strategy = st.radio(
                "Trading Strategy",
                ["Intraday", "Swing Trading"],
                index=1
            )
            refresh_button = st.button("Refresh Data")

        # Fetch stock data
        try:
            with st.spinner("Fetching stock data..."):
                stock_data = yf.download(stock_symbol, period=timeframe)

            if stock_data.empty:
                st.error(f"No data found for stock symbol: {stock_symbol}")
                return

            # Analyze stock
            analyzed_data = self.generate_buy_sell_signals(stock_data)

            # Sentiment Analysis for Stock
            sentiment = self.analyze_sentiment(f"Sentiment about stock {stock_symbol}")

            # Main content columns
            col1, col2 = st.columns([3, 1])

            with col1:
                # Stock Price Chart
                st.plotly_chart(
                    self.plot_stock_data(analyzed_data),
                    use_container_width=True
                )

            with col2:
                # Sentiment and Strategy Insights
                st.header("Insights")

                st.metric("Sentiment Score",
                          f"{sentiment['score']}/100",
                          sentiment['category'])

                # Technical Indicators
                st.subheader("Key Indicators")
                latest_rsi = analyzed_data['RSI'].iloc[-1]
                st.metric("RSI", f"{latest_rsi:.2f}")

                # Buy/Sell Recommendation
                if latest_rsi < 30:
                    st.success("🟢 Buy Signal")
                elif latest_rsi > 70:
                    st.warning("🔴 Sell Signal")
                else:
                    st.info("🟡 Hold Signal")

                # Strategy Recommendation
                st.subheader("Strategy")
                if strategy == "Intraday":
                    st.write("Recommended for same-day trading")
                else:
                    st.write("Recommended for 1-3 months holding")

            # Supplementary Information
            with st.expander("Detailed Technical Analysis"):
                technical_data = analyzed_data[['Close', 'RSI', 'MACD', 'Signal_Line', 'Buy_Signal', 'Sell_Signal']]
                st.write(technical_data)
                csv = technical_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{stock_symbol}_technical_analysis.csv",
                    mime="text/csv",
                )

        except yfinance.exceptions.YFinanceError as e:
            st.error(f"Error fetching data for {stock_symbol}: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")


def main():
    tool = StockTrendAnalysisTool()
    tool.run()


if __name__ == "__main__":
    main()
