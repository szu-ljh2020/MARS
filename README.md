# Reference implementation of our paper A Motif-based Autoregressive Model for Retrosynthesis Prediction

# conda environment
We recommend to new a Conda environment to run the code.

# Step-1: Data Processing
```
python prepare_mol_graph.py
```

# Step-2: Training
```
python run_gnn.py
```

You can setup hyperparameters like:
```
python run_gnn.py --epochs 100 --device 0
```

# Step-3: Testing
```
python run_gnn.py --test_only --input_model_file model_e100.pt
```

you can use multiprocessing to speed up the infernece phase:
```
python run_gnn.py --test_only --input_model_file model_e100.pt --num_process 16
```
