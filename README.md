# spytorch
## Developed from 
- Cadene's [bootstrap.pytorch](https://github.com/Cadene/bootstrap.pytorch) 
- KaihuaTang's [VQA2.0's Recent Approaches](https://github.com/KaihuaTang/VQA2.0-Recent-Approachs-2018.pytorch) 
- Cyanogenoid's [vqa-counting](https://github.com/Cyanogenoid/vqa-counting/tree/master/vqa-v2) 

## Workflow
Most common steps:
- train and eval: select hyperparameters
```bash
# train on train dataset; eval on val dataset
python main.py
```
- train: train on more data
```bash
# train on train and val dataset; --train means no validation during train
python main.py --train --train_split=trainval
```
- val: this step can be skipped
```bash
python main.py --val 
```
- test: test on test dataset if exists
```bash
python main.py --test --test_split=test
```

### Train
```bash
# default train and val (config/config.py)
python main.py

# specify config file (config/config_sgd.py)
python main.py --conf config_sgd

# resume
python main.py --conf config_sgd --resume

# use --train to only train
python main.py --train
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


### Stages Summary
#### train
- dataset: train|trainval
- criterion: loss
- output: model

#### val
- dataset: val|train
- output: pred_label_file, accuracy 

#### test
- dataset: test|val|train
- output: pred_label_file