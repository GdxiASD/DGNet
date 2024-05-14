# DGNet: Dynamic Graph Network for Multivariate Time Series Prediction

### Running the Codes
```
python main.py --data pems03 --feature_size 358 \
--model DGNetv2 --exid best_pems03_0513 \
--in_dim 2 --hidden_size 32 --embed_size 32 --layer 2 --batch_size 64 --blocks 4 \
--TCN --GCN --blocks_gate --graph_regenerate --is_graph_shared --bi

python main.py --data metr --feature_size 207 \
--model DGNetv2 --exid best_metr_0513 \
--in_dim 1 --hidden_size 32 --embed_size 32 --layer 2 --batch_size 64 --blocks 6 \
--TCN --GCN --blocks_gate --graph_regenerate --is_graph_shared --bi

python main.py --data pems07 --feature_size 228 \
--model DGNetv2 --exid best_pems07_0513 \
--in_dim 2 --hidden_size 32 --embed_size 32 --layer 2 --batch_size 64 --blocks 6 \
--TCN --GCN --blocks_gate --graph_regenerate --is_graph_shared --bi

python main.py --data Electricity2 --feature_size 321 \
--model DGNetv2 --exid best_Electricity_0513 \
--in_dim 2 --hidden_size 32 --embed_size 32 --layer 2 --batch_size 64 --blocks 4 \
--TCN --GCN --blocks_gate --graph_regenerate --is_graph_shared --bi

python main.py --data ECG --feature_size 140 \
--model DGNetv2 --exid best_ECG_0513 \
--in_dim 1 --hidden_size 32 --embed_size 32 --layer 2 --batch_size 32 --blocks 6 \
--TCN --GCN --blocks_gate --graph_regenerate --is_graph_shared --bi
```
