# for split data
# nohup ./build/CellTree_split /data/jitao/dataset/OSM_US_NE/20_outliers_lon_lat.csv /data/jitao/dataset/OSM_US_NE/split/ > /data/jitao/dataset/OSM_US_NE/split.log 2>&1 &
# nohup ./build/CellTree_split /data/jitao/dataset/OSM/osm.csv /data/jitao/dataset/OSM/split2/ > /data/jitao/dataset/OSM/split2.log 2>&1 &
# nohup ./build/CellTree_split /data/jitao/dataset/Tiger/center_tiger_east_17m.txt /data/jitao/dataset/Tiger/split/ > /data/jitao/dataset/Tiger/split.log 2>&1 &
# for training 
# nohup ./build/CellTree /data/jitao/dataset/OSM/osm.csv /data/jitao/dataset/OSM/trained_modelParam_for_split2/ > log/pointSearch_v1.3.log 2>&1 &
# change range search
nohup ./build/CellTree /data/jitao/dataset/OSM/osm.csv /data/jitao/dataset/OSM/new_trained_model_param_for_split2/ > log/rangeSearch_v1.2.log 2>&1 &
# nohup ./build/CellTree /data/jitao/dataset/OSM/osm.csv /data/jitao/dataset/OSM/trained_modelParam_for_split2_largeBatch/ > log/rangeSearch_lbatch_v1.1.log 2>&1 &
# nohup ./build/CellTree /data/jitao/dataset/OSM/osm.csv /data/jitao/dataset/OSM/trained_modelParam_for_split2_noac2/ > log/rangeSearch_noac2_v1.1.log 2>&1 &