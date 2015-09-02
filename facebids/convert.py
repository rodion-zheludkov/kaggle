from __future__ import print_function
import pandas as pd


def join_agg_str(df):
    return pd.Series(dict(
        urls=';'.join(set(df['url'].astype(str))),
        auctions=';'.join(set(df['auction'].astype(str))),
        merchandises=';'.join(set(df['merchandise'].astype(str))),
        devices=';'.join(set(df['device'].astype(str))),
        countries=';'.join(set(df['country'].astype(str))),
        ips=';'.join(set(df['ip'].astype(str))),
        payment_account=df['payment_account'].iloc[0],
        address=df['address'].iloc[0],
        outcome=df['outcome'].iloc[0]
    ))


def join_agg_str_no_outcome(df):
    return pd.Series(dict(
        urls=';'.join(set(df['url'].astype(str))),
        auctions=';'.join(set(df['auction'].astype(str))),
        merchandises=';'.join(set(df['merchandise'].astype(str))),
        devices=';'.join(set(df['device'].astype(str))),
        countries=';'.join(set(df['country'].astype(str))),
        ips=';'.join(set(df['ip'].astype(str))),
        payment_account=df['payment_account'].iloc[0],
        address=df['address'].iloc[0],
        outcome=0.
    ))


def join_agg_count(df):
    return pd.Series(dict(
        total=len(df),
        auctions=len(set(df['auction'].astype(str))),
        merchandises=len(set(df['merchandise'].astype(str))),
        devices=len(set(df['device'].astype(str))),
        countries=len(set(df['country'].astype(str))),
        ips=len(set(df['ip'].astype(str))),
        payment_account=df['payment_account'].iloc[0],
        address=df['address'].iloc[0],
        outcome=df['outcome'].iloc[0]
    ))


def agg_total(df):
    return pd.Series(dict(
        bidder_id=df['bidder_id'].iloc[0],
        total=len(df),
        auctions=len(set(df['auction'].astype(str))),
        merchandises=len(set(df['merchandise'].astype(str))),
        devices=len(set(df['device'].astype(str))),
        countries=len(set(df['country'].astype(str))),
        ips=len(set(df['ip'].astype(str))),
        payment_account=df['payment_account'].iloc[0],
        address=df['address'].iloc[0],
        outcome=df['outcome'].iloc[0],
        proba=df['proba'].iloc[0],
        device_per_ip_max=df['device_per_ip'].max(),
        device_per_ip_min=df['device_per_ip'].min(),
        device_per_ip_mean=df['device_per_ip'].mean(),
        device_per_ip_std=df['device_per_ip'].std(),
        device_per_auc_max=df['device_per_auc'].max(),
        device_per_auc_min=df['device_per_auc'].min(),
        device_per_auc_mean=df['device_per_auc'].mean(),
        device_per_auc_std=df['device_per_auc'].std(),
        dt_max=df['dt'].max(),
        dt_min=df['dt'].min(),
        dt_mean=df['dt'].mean(),
        dt_std=df['dt'].std(),
        # bid_dt_max=df['bid_dt'].max(),
        # bid_dt_min=df['bid_dt'].min(),
        # bid_dt_mean=df['bid_dt'].mean(),
        # bid_dt_std=df['bid_dt'].std()
    ))


def agg_auction(df):
    return pd.Series(dict(
        bidder_id=df['bidder_id'].iloc[0],
        auc_total=len(df),
        auc_devices=len(set(df['device'].astype(str))),
        auc_countries=len(set(df['country'].astype(str))),
        auc_ips=len(set(df['ip'].astype(str))),
        auc_dt_max=df['dt'].max(),
        auc_dt_min=df['dt'].min(),
        auc_dt_mean=df['dt'].mean(),
        auc_dt_std=df['dt'].std(),
        # auc_bid_dt_max=df['bid_dt'].max(),
        # auc_bid_dt_min=df['bid_dt'].min(),
        # auc_bid_dt_mean=df['bid_dt'].mean(),
        # auc_bid_dt_std=df['bid_dt'].std()
    ))


def agg_auction_mean(df):
    return pd.Series(dict(
        bidder_id=df['bidder_id'].iloc[0],
        auc_total=df['auc_total'].mean(),
        auc_devices=df['auc_devices'].mean(),
        auc_countries=df['auc_countries'].mean(),
        auc_ips=df['auc_ips'].mean(),
        auc_dt_max=df['auc_dt_max'].mean(),
        auc_dt_min=df['auc_dt_min'].mean(),
        auc_dt_mean=df['auc_dt_mean'].mean(),
        auc_dt_std=df['auc_dt_std'].mean(),
        # auc_bid_dt_max=df['auc_bid_dt_max'].mean(),
        # auc_bid_dt_min=df['auc_bid_dt_min'].mean(),
        # auc_bid_dt_mean=df['auc_bid_dt_mean'].mean(),
        # auc_bid_dt_std=df['auc_bid_dt_std'].mean()
    ))


def join(out, input_file, join_func):
    tr = pd.read_csv(input_file, index_col='bidder_id')
    bids = pd.read_csv('/home/rodion/facebids/bids.csv', index_col='bidder_id')
    joined = tr.join(bids, how='inner')
    joined.drop(['bid_id', 'time'], axis=1, inplace=True)

    joined = joined.reset_index()

    joined_agg = joined.groupby('bidder_id').apply(join_func)
    joined_agg.to_csv(out)


def join_with_proba(out):
    tr = pd.read_csv('/home/rodion/facebids/train.csv', index_col='bidder_id')
    bids = pd.read_csv('/home/rodion/facebids/bids.csv', index_col='bidder_id')
    proba = pd.read_csv('/home/rodion/facebids/train/vw.proba.csv', index_col='bidder_id')
    print('read csv done')

    bids.reset_index(inplace=True)
    bids['dt'] = bids.groupby(['bidder_id', 'auction'])['time'].transform(pd.Series.diff)
    bids['dt'] = bids['dt'].fillna(bids['dt'].mean())
    # bids['bid_dt'] = bids.groupby(['auction'])['time'].transform(pd.Series.diff)
    # bids['bid_dt'] = bids['bid_dt'].fillna(bids['bid_dt'].mean())
    print('dt done')

    ipdev_df = bids.groupby(['bidder_id', 'ip'])['device'].apply(lambda x: len(set(x.astype(str)))).reset_index()
    ipdev_df = ipdev_df.rename(columns={'device': 'device_per_ip'})

    aucdev_df = bids.groupby(['bidder_id', 'auction'])['device'].apply(lambda x: len(set(x.astype(str)))).reset_index()
    aucdev_df = aucdev_df.rename(columns={'device': 'device_per_auc'})

    bids = pd.merge(bids, ipdev_df, on=['bidder_id', 'ip'])
    bids = pd.merge(bids, aucdev_df, on=['bidder_id', 'auction'])
    print('ipdev done')

    bids.set_index('bidder_id', inplace=True)
    joined = tr.join(bids, how='inner').join(proba, how='inner')
    joined.drop(['bid_id', 'time'], axis=1, inplace=True)
    joined.reset_index(inplace=True)
    print('join done')

    joined_agg_total = joined.groupby('bidder_id').apply(agg_total)
    joined_agg_total.set_index('bidder_id', inplace=True)
    joined_agg_auction = joined.groupby(['bidder_id', 'auction']).apply(agg_auction)
    joined_agg_auction_mean = joined_agg_auction.groupby('bidder_id').apply(agg_auction_mean)
    joined_agg_auction_mean.set_index('bidder_id', inplace=True)
    print('agg done')

    res = joined_agg_total.join(joined_agg_auction_mean, how='inner')
    res['dt_std'] = res['dt_std'].fillna(res['dt_std'].mean())
    res['auc_dt_std'] = res['auc_dt_std'].fillna(res['auc_dt_std'].mean())
    res['device_per_ip_std'] = res['device_per_ip_std'].fillna(res['device_per_ip_std'].mean())
    res['device_per_auc_std'] = res['device_per_auc_std'].fillna(res['device_per_auc_std'].mean())
    # res['bid_dt_std'] = res['bid_dt_std'].fillna(res['bid_dt_std'].mean())
    # res['auc_bid_dt_std'] = res['auc_bid_dt_std'].fillna(res['auc_bid_dt_std'].mean())
    print('fillna done')

    res.to_csv(out)


def _as_vw_string(x, tag=None, y=None, weights=None):
    if y:
        result = str(y) + ' '
    else:
        result = "1 "

    if weights:
        result += str(weights[y - 1]) + ' '

    if tag:
        result += "'" + tag

    for namespace_name, namespace in x.items():
        namespace_str = " ".join(["%s" % key for (key, value) in namespace.items()])
        result += '|' + namespace_name + ' ' + namespace_str + ' '

    return result


def convert_join_row(row):
    auction = {x: 1. for x in row[2].split(';')}
    country = {x: 1. for x in row[3].split(';')}
    device = {x: 1. for x in row[4].split(';')}
    ip = {x: 1. for x in row[5].split(';')}
    merchandise = {x: 1. for y in row[6].split(';') for x in y.split(' ')}
    url = {x: 1. for x in row[9].split(';')}
    outcome = row[7]
    bidder_id = row[0]

    return {'a': ip, 'b': auction, 'c': merchandise, 'd': device, 'e': country, 'u': url}, bidder_id, \
           int(float(outcome)) + 1


def convert_csv_vw(filename, convert_row):
    with open(filename, 'r') as f, open(filename + '.vw', 'w') as fw:
        for _ in f:
            break

        c = 0
        for row in f:
            c += 1
            if c % 10000 == 0:
                print('Done ', c)
            row = row.strip().split(',')
            x, tag, y = convert_row(row)
            fw.write(_as_vw_string(x, tag, y) + '\n')


if __name__ == '__main__':
    train_csv_1 = '/home/rodion/facebids/train/join.str.csv'
    train_csv_2 = '/home/rodion/facebids/train/join.count.csv'
    train_csv_3 = '/home/rodion/facebids/train/join.count.proba.csv'
    train_input = '/home/rodion/facebids/train.csv'

    # join(train_csv_1, train_input, join_agg_str)
    # convert_csv_vw(train_csv_1, convert_join_row)

    # join(joined_csv_2, train_input, join_agg_count)

    # join_with_proba(joined_csv_3)

    test_csv_1 = '/home/rodion/facebids/test/join.str.csv'
    test_input = '/home/rodion/facebids/test.csv'

    join(test_csv_1, test_input, join_agg_str_no_outcome)
    convert_csv_vw(test_csv_1, convert_join_row)