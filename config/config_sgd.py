# exp
exp_dir = 'logs/mnist/default'

# dataset:
dataset_name = 'datasets.mnist'
dir_data = 'data/mnist'
nb_threads = 4
batch_size = 64
pin_memory = False

# model:
model_name = 'models.SimpleNet'
criterion_name = 'nll'

# optimizer:
optim_method = 'sgd'

lr=0.01
momentum=0.5

#schedule_method = 'warm_up'
schedule_method = 'batch_decay'

gradual_warmup_steps = [2.0 * lr, 2.0 * lr, 1.0 * lr, 1.0 * lr]
lr_decay_step = 2
lr_decay_rate = 0.25
lr_halflife = 938 # for scheduler (counting) 50000
lr_decay_epochs = range(10, 100, lr_decay_step)


# metric
metric_topk=[1,5]

# engine:
name = 'default'
nb_epochs=6
debug = False
print_freq = 10

# misc:
seed = 1337

# views: