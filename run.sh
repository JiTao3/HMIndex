# for split data
# nohup ./build/CellTree_split /data/jitao/dataset/OSM/osm.csv /data/jitao/dataset/OSM/split2/ > /data/jitao/dataset/OSM/split2.log 2>&1 &
# nohup ./build/CellTree_split /data/jitao/dataset/Tiger/center_tiger_east_17m.txt /data/jitao/dataset/Tiger/split/ > /data/jitao/dataset/Tiger/split.log 2>&1 &
# nohup ./build/CellTree_split /data/jitao/dataset/uniform/2d_len_1e8_seed_1.csv /data/jitao/dataset/uniform/split/ > /data/jitao/dataset/uniform/split.log 2>&1 &
# nohup ./build/CellTree_split /data/jitao/dataset/skewed/2d_len_1e8_seed_1.csv /data/jitao/dataset/skewed/split/ > /data/jitao/dataset/skewed/split.log 2>&1 &

# for training 
# nohup ./build/CellTree /data/jitao/dataset/OSM/osm.csv /data/jitao/dataset/OSM/trained_modelParam_for_split2/ > log/pointSearch_v1.3.log 2>&1 &
# change range search
# nohup ./build/CellTree /data/jitao/dataset/OSM/osm.csv /data/jitao/dataset/OSM/new_trained_model_param_for_split2/ > log/rangeSearch_v1.2.log 2>&1 &
# nohup ./build/CellTree /data/jitao/dataset/OSM/osm.csv /data/jitao/dataset/OSM/trained_modelParam_for_split2_largeBatch/ > log/rangeSearch_lbatch_v1.1.log 2>&1 &
# nohup ./build/CellTree /data/jitao/dataset/OSM/osm.csv /data/jitao/dataset/OSM/trained_modelParam_for_split2_noac2/ > log/rangeSearch_noac2_v1.1.log 2>&1 &

# * point search
# nohup ./build/CellTree /data/jitao/dataset/OSM/osm.csv /data/jitao/dataset/OSM/point_query_sample_10w.csv /data/jitao/dataset/OSM/trained_modelParam_for_split2/ > log/osm_cn_v2.log 2>&1 &
# nohup ./build/CellTree /data/jitao/dataset/OSM_US_NE/20_outliers_lon_lat.csv /data/jitao/dataset/OSM_US_NE/point_query_sample_10w.csv /data/jitao/dataset/OSM_US_NE/trained_modelParam_for_split/ > log/osm_us_ne_pointv3.log 2>&1 &
# nohup ./build/CellTree /data/jitao/dataset/uniform/2d_len_1e8_seed_1.csv /data/jitao/dataset/uniform/point_query_sample_10w.csv /data/jitao/dataset/uniform/trained_modelParam_for_split/ > log/uniform_v3.log 2>&1 &
nohup ./build/CellTree /data/jitao/dataset/skewed/2d_len_1e8_seed_1.csv /data/jitao/dataset/skewed/point_query_sample_10w.csv /data/jitao/dataset/skewed/trained_modelParam_for_split/ > log/skewed_fix_v8.log 2>&1 &
# nohup ./build/CellTree /data/jitao/dataset/Tiger/center_tiger_east_17m.txt /data/jitao/dataset/Tiger/point_query_sample_10w.csv /data/jitao/dataset/Tiger/trained_modelParam_for_split/ > log/tiger_point_v3.log 2>&1 &
