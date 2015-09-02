from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt


def describe_both(dfs):
    for df in dfs:
        # Auction per user
        df.groupby(['bidder_id'])['auction'].nunique().describe()
        # Bids per user
        df.groupby(['bidder_id']).size().describe()


def plot(dfs):
    # plt.boxplot([df.groupby(['bidder_id'])['auction'].nunique() for df in dfs])
    plt.boxplot([df.groupby(['bidder_id']).size() for df in dfs])
    plt.show()


def describe(csv):
    data = pd.read_csv(csv)
    # describe_both([data, data[data['outcome'] == 0.], data[data['outcome'] == 1.]])
    plot([data, data[data['outcome'] == 0.], data[data['outcome'] == 1.]])


def ts_func(df):
    return pd.Series(dict(
        dt=df['time'].diff()
    ))


def ts(csv):
    df = pd.read_csv(csv)
    df['dt'] = df.groupby(['bidder_id', 'auction'])['time'].transform(pd.Series.diff)
    df['dt'] = df['dt'].fillna(df['dt'].mean())


if __name__ == '__main__':
    ts('/home/rodion/facebids/bids.csv')