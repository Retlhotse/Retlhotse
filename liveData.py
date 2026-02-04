import yfinance as yf
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import datetime

# Define the currency pair
symbol = "EURGBP=X"


def fetch_data():
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d", interval='1m')
    return data


# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Live EUR/GBP Forex Data"),
    dcc.Graph(id='forex-graph'),
    dcc.Interval(
        id='interval-component',
        interval=60 * 1000,  # Update every minute
        n_intervals=0
    )
])


@app.callback(
    Output('forex-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    data = fetch_data()
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Market Data'
    ))

    fig.update_layout(
        title='EURGBP Live Forex Data',
        yaxis_title='Currency Pair Price',
        xaxis_rangeslider_visible=True
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
