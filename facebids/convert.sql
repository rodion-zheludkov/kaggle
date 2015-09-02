create table bids (bidder_id char(37), auction char(5), ip varchar(14), bid_id int, merchandise varchar(100), device varchar(10), time long, country char(2), url char(15), dt long, a_dt long, ai_dt long);

load data infile '/home/rodion/facebids/bids_dt.csv' into table rodion.bids FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 ROWS;

create table bids_ai select bidder_id, auction, ip, max(ai_dt) as ai_dt_max, min(ai_dt) as ai_dt_min, avg(ai_dt) as ai_dt_avg, std(ai_dt) as ai_dt_std from bids where ai_dt <> '' group by bidder_id, auction, ip;

create table bids_a select bidder_id, auction, max(a_dt) as a_dt_max, min(a_dt) as a_dt_min, avg(a_dt) as a_dt_avg, std(a_dt) as a_dt_std from bids where a_dt <> '' group by bidder_id, auction;

create table bids_b select bidder_id, max(dt) as dt_max, min(dt) as dt_min, avg(dt) as dt_avg, std(dt) as dt_std from bids where dt <> '' group by bidder_id;

create table bids_ai_b select bidder_id, avg(ai_dt_max) as ai_dt_max, avg(ai_dt_min) as ai_dt_min, avg(ai_dt_avg) as ai_dt_avg, avg(ai_dt_std) as ai_dt_std from bids_ai group by bidder_id;

drop table bids_ai_b;
create table bids_ai_b select bidder_id, max(ai_dt_max) as ai_dt_max, min(ai_dt_min) as ai_dt_min, avg(ai_dt_avg) as ai_dt_avg, min(ai_dt_std) as ai_dt_std_min, max(ai_dt_std) as ai_dt_std_max, avg(ai_dt_std) as ai_dt_std from bids_ai group by bidder_id;

create table bids_a_b select bidder_id, avg(a_dt_max) as a_dt_max, avg(a_dt_min) as a_dt_min, avg(a_dt_avg) as a_dt_avg, avg(a_dt_std) as a_dt_std from bids_a group by bidder_id;

drop table bids_a_b;
create table bids_a_b select bidder_id, max(a_dt_max) as a_dt_max, min(a_dt_min) as a_dt_min, avg(a_dt_avg) as a_dt_avg, min(a_dt_std) as a_dt_std_min, max(a_dt_std) as a_dt_std_max, avg(a_dt_std) as a_dt_std from bids_a group by bidder_id;


create table bids_ai_cnt select bidder_id, auction, ip, count(1) as ai_total, count(distinct(device)) as ai_d_total from bids group by bidder_id, auction, ip;

create table bids_a_cnt select bidder_id, auction, count(1) as a_total, count(distinct(device)) as a_d_total, count(distinct(ip)) as a_i_total, count(distinct(country)) as a_c_total from bids group by bidder_id, auction;

create table bids_b_cnt select bidder_id, count(1) as b_total, count(distinct(device)) as b_d_total, count(distinct(ip)) as b_i_total, count(distinct(country)) as b_c_total, count(distinct(auction)) as b_a_total from bids group by bidder_id;


create table bids_ai_b_cnt select bidder_id, avg(ai_total) as ai_total, avg(ai_d_total) as ai_d_total from bids_ai_cnt group by bidder_id;

create table bids_a_b_cnt select bidder_id, avg(a_total) as a_total, avg(a_d_total) as a_d_total, avg(a_i_total) as a_i_total, avg(a_c_total) as a_c_total from bids_a_cnt group by bidder_id;

drop table bids_new;
create table bids_new select 
t0.bidder_id, t0.b_total, t0.b_d_total, t0.b_i_total, t0.b_c_total, 
t1.ai_dt_max, t1.ai_dt_min, t1.ai_dt_avg, t1.ai_dt_std, t1.ai_dt_std_min, t1.ai_dt_std_max, 
t2.a_dt_max, t2.a_dt_min, t2.a_dt_avg, t2.a_dt_std, t2.a_dt_std_min, t2.a_dt_std_max,
t3.dt_max, t3.dt_min, t3.dt_avg, t3.dt_std,
t4.ai_total, t4.ai_d_total,
t5.a_total, t5.a_d_total, t5.a_i_total, t5.a_c_total
from bids_b_cnt t0 
left join bids_ai_b t1 on t0.bidder_id = t1.bidder_id 
left join bids_a_b t2 on t0.bidder_id = t2.bidder_id 
left join bids_b t3 on t0.bidder_id = t3.bidder_id 
left join bids_ai_b_cnt t4 on t0.bidder_id = t4.bidder_id 
left join bids_a_b_cnt t5 on t0.bidder_id = t5.bidder_id;

SELECT *
FROM bids_new
INTO OUTFILE '/tmp/bids_new.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';


bidder_id,b_total,b_d_total,b_i_total,b_c_total,ai_dt_max,ai_dt_min,ai_dt_avg,ai_dt_std,ai_dt_std_min,ai_dt_std_max,a_dt_max,a_dt_min,a_dt_avg,a_dt_std,a_dt_std_min,a_dt_std_max,dt_max,dt_min,dt_avg,dt_std,ai_total,ai_d_total,a_total,a_d_total,a_i_total,a_c_total 