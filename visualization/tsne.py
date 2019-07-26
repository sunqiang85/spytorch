import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from tqdm import tqdm


class LayerOutNet(nn.Module):
    def __init__(self, model, layer):
        super(LayerOutNet, self).__init__()
        self.model = model
        self.layer = layer

        def save_output(module, input, output):
            self.buffer = output
        self.layer.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer



def gen_output_features(model, val_loader):
    tq = tqdm(val_loader, desc='{}'.format('val'), ncols=100)
    model.eval()
    outputs=[]
    targets=[]
    with torch.no_grad():
        i=j=0
        for batch_idx, item in enumerate(tq):
            data = item['data']
            target = item['class_id'].squeeze(1)
            data, target = data.cuda(), target.cuda()
            output = model(data)
            outputs.append(output.cpu())
            targets.append(target.cpu())
    out_features = torch.cat(outputs, dim=0)
    out_targets = torch.cat(targets, dim=0)
    return out_features.numpy(), out_targets.numpy()


# API for Visualzation
## save output features
def save_outfeature(config, model, val_loader):
    out_features_path = os.path.join(config.exp_dir, 'fc2.npy')
    out_targets_path = os.path.join(config.exp_dir, 'targets.npy')

    layeroutmodel = LayerOutNet(model,model.module.fc2)
    out_features, out_targets = gen_output_features(model=layeroutmodel, val_loader=val_loader)

    np.save(out_features_path,out_features)
    np.save(out_targets_path, out_targets)

def save_figure(config):
    out_features_path = os.path.join(config.exp_dir, 'fc2.npy')
    out_targets_path = os.path.join(config.exp_dir, 'targets.npy')
    fig_path = os.path.join(config.exp_dir, 'tsne.png')
    # load data
    X = np.load(out_features_path)
    y=np.load(out_targets_path)


    # fit
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)
    plt.figure(figsize=(6, 5))

    # figure config
    target_ids = [c for c in range(10)]
    target_names = [str(c) for c in range(10)]
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    k=20
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(X_2d[y == i, 0][:k], X_2d[y == i, 1][:k], c=c, label=label)
    plt.legend()
    plt.savefig(fig_path)
    plt.savefig('figs/mnist/tsne.png')
    plt.show()


