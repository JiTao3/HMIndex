# python train.py osm_ne_us >> ../log/train_random/osm_ne_us_no_db.log && \ 
# python train.py tiger >> ../log/train_random/tiger_no_db.log && \ 
# python train.py uniform >> ../log/train_random/uniform_no_db.log && \ 
# python train.py skewed >> ../log/train_random/skewed_no_db.log


# python utils.py /data/jitao/dataset/Tiger/split/ /home/jitao/leo_index_pointnet/log/tiger/tiger_1.pt /data/jitao/dataset/Tiger/initial_param_mono/ ../log/initial_param_mono_tiger.log && \
# python utils.py /data/jitao/dataset/OSM_US_NE/split/ /home/jitao/leo_index_pointnet/log/osm_en_us/osm_ne_us_1.pt /data/jitao/dataset/OSM_US_NE/initial_param_mono/ ../log/initial_param_mono_OSM_US_NE.log && \
# python utils.py /data/jitao/dataset/uniform/split/ /home/jitao/leo_index_pointnet/log/uniform/uniform_1.pt /data/jitao/dataset/uniform/initial_param_mono/ ../log/initial_param_mono_uniform.log && \
# python utils.py /data/jitao/dataset/skewed/split/ /home/jitao/leo_index_pointnet/log/skewed/skewed_1.pt /data/jitao/dataset/skewed/initial_param_mono/ ../log/initial_param_mono_skewed.log 


python MonoMLP.py osm_ne_us >> ../log/train_log/osm_mono.log && \
python MonoMLP.py tiger >> ../log/train_log/tiger_mono.log && \
python MonoMLP.py uniform >> ../log/train_log/uniform_mono.log && \
python MonoMLP.py skewed >> ../log/train_log/skewed_mono.log 


# python MonoMLP.py osm_ne_us >> ../log/train_log/osm_error.log && \
# python MonoMLP.py tiger >> ../log/train_log/tiger_error.log && \
# python MonoMLP.py uniform >> ../log/train_log/uniform_error.log && \
# python MonoMLP.py skewed >> ../log/train_log/skewed_error.log 