# Created by cc215 at 27/12/19
# Enter feature description here
# Enter scenario name here
# Enter steps here
import json
import logging
import os
import shutil

import torch
import matplotlib.pyplot as plt


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def plot_training_results(model_dir, plot_history):
    """
    Plot training results (procedure) during training.
    Args:
        plot_history: (dict) a dictionary containing historical values of what
                      we want to plot
    """
    # tr_losses = plot_history['train_loss']
    # val_losses = plot_history['val_loss']
    # te_losses = plot_history['test_loss']
    # tr_accs = plot_history['train_acc']
    val_accs = plot_history['val_acc']
    te_accs = plot_history['test_acc']

    # plt.figure(0)
    # plt.plot(list(range(len(tr_losses))), tr_losses, label='train_loss')
    # plt.plot(list(range(len(val_losses))), val_losses, label='val_loss')
    # plt.plot(list(range(len(te_losses))), te_losses, label='test_loss')
    # plt.title('Loss trend')
    # plt.xlabel('episode')
    # plt.ylabel('ce loss')
    # plt.legend()
    # plt.savefig(os.path.join(model_dir, 'loss_trend'), dpi=200)
    # plt.clf()

    plt.figure(1)
    # plt.plot(list(range(len(tr_accs))), tr_accs, label='train_acc')
    plt.plot(list(range(len(val_accs))), val_accs, label='val_acc')
    plt.plot(list(range(len(te_accs))), te_accs, label='test_acc')
    plt.title('Accuracy trend')
    plt.xlabel('iter / 1000')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'accuracy_trend'), dpi=200)
    plt.clf()


# if __name__ =='__main__':
#     params = Params('/vol/medic01/users/cc215/Dropbox/projects/DeformADA/configs/gat_loss.json')
#     print (params.dict)
