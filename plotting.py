import matplotlib.pyplot as plt
import torch

def plot_accuracy(acc_hist, title, test=False):
    fig, ax = plt.subplots()
    ax.plot(acc_hist)
    ax.set_title(title)
    ax.set_xlabel("Epoch" if test else "Batch")
    ax.set_ylabel("Accuracy")
    fig.savefig("images/"+title+"_ISING.svg", format='svg')

def plot_loss(loss_hist, title):
    fig, ax = plt.subplots()
    ax.plot(loss_hist)
    ax.set_title(title)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    fig.savefig("images/"+title+"_ISING.svg", format='svg')

def plot_comparison(**test_acc):
    fig, ax = plt.subplots()
    for key, val in test_acc.items():
        ax.plot(val, label=key)

    ax.set_title("Test-accuracy comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.savefig("images/Test-accuracy_Comparison_ISING.svg", format='svg')

def plot_predictions(Ts, **predictions):
    """ Predictions in the form of `key={val: predictions, color: 'color_name'}` """
    fig, ax = plt.subplots()
    labels = []
    lines = []
    for key, val in predictions.items():
        labels.append(key)
        lines.append(ax.plot(Ts, val['val'], color=val['color']))

    ax.set_title("Tc predictions comparison")
    ax.set_xlabel("T")
    ax.set_ylabel("Predictions")
    ax.legend(handles=[line[0] for line in lines],labels=labels)

    fig.savefig("images/Tc_predictions_Comparison.svg", format='svg')

def plot_predicted_Tc(x, y_rate, y_latency, xlabel=""):
    Tc = 2.2692
    fig, ax = plt.subplots()
    ax.set_title(f"Tc prediction per {xlabel}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Predicted Tc")
    ax.plot(x, y_rate, label="Rate coded")
    ax.plot(x, y_latency, label="Latency coded")
    ax.axhline(Tc, color='r', linestyle='--', label='Tc')
    ax.legend()
    fig.savefig(f"images/Tc_predictions_{xlabel}.svg", format='svg')