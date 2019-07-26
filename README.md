# spytorch
## Developed from 
- Cadene's [bootstrap.pytorch](https://github.com/Cadene/bootstrap.pytorch)
- KaihuaTang's [VQA2.0's Recent Approaches](https://github.com/KaihuaTang/VQA2.0-Recent-Approachs-2018.pytorch)

## Workflow
### Train
```python
# default (config/config.py)
python main.py

# specify config file (config/config_sgd.py)
python main.py --conf config_sgd

# resume
python main.py --conf config_sgd --resume

```

### Evaluation
```python 
# only evaluation default (config/config.py)
python main.py --eval

# only evaluation
python main.py --conf config_sgd --eval

```

## Docs for Program
### argparse & config
- usiang argparse to load specify conf
```python
# main.py
config=importlib.import_module("config."+args.conf)
# config/config.py
lr = 0.01
...
```

### Dataset Configuration
```python
# main.py
dataset_module = importlib.import_module(config.dataset_name)
# config/config.py: dataset
dataset_name = 'datasets.mnist'
dir_data = 'data/mnist'
nb_threads = 4
batch_size = 64
pin_memory = False
```

### Model Configuration
```python
# main.py
net_module = importlib.import_module(config.model_name)
# config/config.py: model
model_name = 'models.SimpleNet'
criterion_name = 'nll'
```

### Evaulation
configure plots during evaluation, you can add more visualization figures
```python
# main.py
for plotname in config.plot_names:
    plotmodule=importlib.import_module(plotname)
    plotmodule.save_outfeature(config, model, val_loader)
    plotmodule.save_figure(config)
    
# config/config.py: eval
## visualization in eval
eval_visualize = True  # save output feature for visualize. e.g., tsne
plot_names  = ['visualization.tsne'] # plots

```

### Others
- Criterion/Loss
- Metrics
- Optimizer
