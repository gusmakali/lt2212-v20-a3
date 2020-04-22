from a3_model import build_train_model 
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve 
import argparse

def plotting(csv, size, output):
    colors = ['g','b','r','y','c']
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')

    for i,s in enumerate(size):
        scores = build_train_model(csv, s, 1, False)
        
        true = scores[-1]
        pred = scores[-2]
        
        curve_v = precision_recall_curve(true, pred)
        prec = curve_v[0]
        recall = curve_v[1]

        plt.plot(recall, prec, colors[i], label=f"hidden layer size {s}")
 
    plt.legend(loc="upper right")
    plt.savefig(output, format='png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot recall/precision curves of different size hidden layers")
    parser.add_argument("inputfile", type=str, help="The output.csv file from part1.")
    parser.add_argument("outputfile", type=str, help="The png file to which to draw the plot.")

    args = parser.parse_args()

    print("Writing graph to plot.png...")
    plotting(args.inputfile, [64, 128, 300, 400, 500], args.outputfile) 
