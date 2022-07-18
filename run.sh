# for split data
# nohup ./build/CellTree_split /data/userpath/dataset/OSM/osm.csv /data/userpath/dataset/OSM/split2/ > /data/userpath/dataset/OSM/split2.log 2>&1 &
# nohup ./build/CellTree_split /data/userpath/dataset/Tiger/center_tiger_east_17m.txt /data/userpath/dataset/Tiger/split2/ > log/tiger_split2.log 2>&1 &
# nohup ./build/CellTree_split /data/userpath/dataset/uniform/2d_len_1e8_seed_1.csv /data/userpath/dataset/uniform/split/ > /data/userpath/dataset/uniform/split.log 2>&1 &
# nohup ./build/CellTree_split /data/userpath/dataset/skewed/2d_len_1e8_seed_1.csv /data/userpath/dataset/skewed/split/ > /data/userpath/dataset/skewed/split.log 2>&1 &

# for training 
# nohup ./build/CellTree /data/userpath/dataset/OSM/osm.csv /data/userpath/dataset/OSM/trained_modelParam_for_split2/ > log/pointSearch_v1.3.log 2>&1 &
# change range search
# nohup ./build/CellTree /data/userpath/dataset/OSM/osm.csv /data/userpath/dataset/OSM/new_trained_model_param_for_split2/ > log/rangeSearch_v1.2.log 2>&1 &
# nohup ./build/CellTree /data/userpath/dataset/OSM/osm.csv /data/userpath/dataset/OSM/trained_modelParam_for_split2_largeBatch/ > log/rangeSearch_lbatch_v1.1.log 2>&1 &
# nohup ./build/CellTree /data/userpath/dataset/OSM/osm.csv /data/userpath/dataset/OSM/trained_modelParam_for_split2_noac2/ > log/rangeSearch_noac2_v1.1.log 2>&1 &

# * point search

# ./build/CellTree uniform >> log/point/no_db_uniform_v1.log  && \
# ./build/CellTree skewed >> log/point/no_db_skewed.log && \
# ./build/CellTree osm_ne_us > log/point/no_db_osm_ne_us_v1.log  && \
# ./build/CellTree tiger > log/point/no_db_tiger_v1.log 

# ./build/CellTree uniform >> log/range/no_db_uniform_v2.log  && \
# ./build/CellTree skewed >> log/range/no_db_skewedv2.log
# ./build/CellTree osm_ne_us >> log/range/no_db_osm_ne_us_v2.log
# ./build/CellTree tiger >> log/range/no_db_tiger_v2.log 


# ./build/CellTree uniform >> log/knn/no_db_uniform_v4.log  && \
# ./build/CellTree skewed >> log/knn/no_db_skewedv4.log && \
# ./build/CellTree osm_ne_us >> log/knn/no_db_osm_ne_us_v4.log && \
./build/CellTree tiger >> log/knn/no_db_tiger_v4.log 


# ./build/CellTree uniform >> log/train_random/random_uniform_v1.log  && \
# ./build/CellTree skewed >> log/train_random/random_skewed.log && \
# ./build/CellTree osm_ne_us >> log/train_random/random_osm_ne_us_v1.log
# ./build/CellTree tiger >> log/train_random/random_tiger_v1.log 


# ./build/CellTree_split uniform >> log/split/uniform_v1.log  && \
# ./build/CellTree_split skewed >> log/split/skewed.log && \
# ./build/CellTree_split osm_ne_us >> log/split/osm_ne_us_v1.log && \
# ./build/CellTree_split tiger >> log/split/tiger_v1.log 

# * rang search
# * windows size [0.0001, 0.0025, 0.01, 0.04, 0.16]
# * aspect raido [0.25, 0.5, 1, 2, 4 ]
# !!! 实验顺序：先遍历window size 再遍历 aspect radio
# nohup ./build/CellTree osm_cn 0.16 0.25 >> log/range/osm_cn_v1.log 2>&1 &
# nohup ./build/CellTree uniform 0.16 0.25 >> log/range/uniform_v1.log 2>&1 &
# nohup ./build/CellTree skewed 0.16 0.25 >> log/range/skewed.log 2>&1 &
# nohup ./build/CellTree osm_ne_us 0.16 0.25 >> log/range/osm_ne_us_v1.log 2>&1 &
# nohup ./build/CellTree tiger 0.16 0.25 >> log/range/tiger_v1.log 2>&1 &

# ./build/CellTree osm_cn 0.0001 4 >> log/range/osm_cn_v1.log  && \
# ./build/CellTree uniform 0.0001 4 >> log/range/uniform_v1.log  && \
# ./build/CellTree skewed 0.0001 4 >> log/range/skewed.log  && \
# ./build/CellTree osm_ne_us 0.0001 4 >> log/range/osm_ne_us_v1.log  && \
# ./build/CellTree tiger 0.0001 4 >> log/range/tiger_v1.log && \

# ./build/CellTree osm_cn 0.0025 4 >> log/range/osm_cn_v1.log  && \
# ./build/CellTree uniform 0.0025 4 >> log/range/uniform_v1.log  && \
# ./build/CellTree skewed 0.0025 4 >> log/range/skewed.log  && \
# ./build/CellTree osm_ne_us 0.0025 4 >> log/range/osm_ne_us_v1.log  && \
# ./build/CellTree tiger 0.0025 4 >> log/range/tiger_v1.log && \

# ./build/CellTree osm_cn 0.01 4 >> log/range/osm_cn_v1.log  && \
# ./build/CellTree uniform 0.01 4 >> log/range/uniform_v1.log  && \
# ./build/CellTree skewed 0.01 4 >> log/range/skewed.log  && \
# ./build/CellTree osm_ne_us 0.01 4 >> log/range/osm_ne_us_v1.log  && \
# ./build/CellTree tiger 0.01 4 >> log/range/tiger_v1.log && \

# ./build/CellTree osm_cn 0.04 4 >> log/range/osm_cn_v1.log  && \
# ./build/CellTree uniform 0.04 4 >> log/range/uniform_v1.log  && \
# ./build/CellTree skewed 0.04 4 >> log/range/skewed.log  && \
# ./build/CellTree osm_ne_us 0.04 4 >> log/range/osm_ne_us_v1.log  && \
# ./build/CellTree tiger 0.04 4 >> log/range/tiger_v1.log && \

# ./build/CellTree osm_cn  >> log/range/v6osm_cn.log 
# ./build/CellTree uniform  >> log/range/v6uniform.log  && \
# ./build/CellTree skewed  >> log/range/v6skewed.log 
# ./build/CellTree osm_ne_us  >> log/range/v6osm_ne_us.log 
# ./build/CellTree tiger >> log/range/v6tiger.log 
# 

# ./build/CellTree osm_cn >> log/knn/osm_cn_v1.log  && \
# ./build/CellTree tiger >> log/knn/v3tiger_v1.log && \
# ./build/CellTree uniform >> log/knn/v3uniform_v1.log  && \
# ./build/CellTree skewed >> log/knn/v3skewed.log && \
# ./build/CellTree osm_ne_us >> log/knn/v3osm_ne_us_v1.log


#  ./build/CellTree >> log/insert/v2skewed_insert.log