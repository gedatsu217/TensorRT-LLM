#!/usr/bin/python3

import sys
import matplotlib.pyplot as plt

data1={}
data2={}

def main():
    global batch_initial, batch_max, output_initial, output_max, tmp1_dir, tmp2_dir
    args = sys.argv
    if len(args) != 7:
        print("Usage: python3 plot_llama.py <batch_intial> <batch_max> <output_initial> <output_max> <tmp1_dir> <tmp2_dir>")
        exit(1)
    
    batch_initial = int(args[1])
    batch_max = int(args[2])
    output_initial = int(args[3])
    output_max = int(args[4])
    tmp1_dir = args[5]
    tmp2_dir = args[6]

    batch = batch_initial
    while batch <= batch_max:
        output = output_initial
        while output <= output_max:
            data1_collect("{}/manifold_llama-{}-{}.log".format(tmp1_dir, batch, output), batch, output)
            data2_collect("{}/manifold_llama-{}-{}.log".format(tmp2_dir, batch, output), batch, output)
            output *= 2
        batch *= 2

    overall_plot()
                
def data1_collect(filename, batch, output): # for tensorrt-llm
    time = 0.0
    with open(filename) as f:
        for line in f:
            if "Time: " in line:
                time += float(line.split()[-2])
    
    data1[(batch, output)] = time/4.0

def data2_collect(filename, batch, output): # for fastertransformer
    time = 0.0
    with open(filename) as f:
        for line in f:
            if "FT-CPP-decoding-beamsearch-time" in line:
                time += float(line.split()[-2])
    
    data2[(batch, output)] = time/4.0

def overall_plot():
    batch = batch_initial
    x = []
    y = []
    y1 = []
    y2 = []
    while batch <= batch_max:
        output = output_initial
        if not batch in [1, 4, 16, 64]:
            batch *= 2
            continue
        while output <= output_max:
            if not output in [2, 8, 32, 128, 512]:
                output *= 2
                continue
            
            x.append((batch, output))
            y1.append(data1[(batch, output)])
            y2.append(data2[(batch, output)])
            #y.append((data2[(batch, output)]-data1[(batch, output)])/data2[(batch, output)]*100)
            output *= 2
        
        batch *= 2

    plt.figure()
    plt.xlabel("(batch size, output size)")
    plt.ylabel("Execution time (ms)")
    label_x = ["({}, {})".format(i[0], i[1]) for i in x]
    plt.xticks([i for i in range(len(x))], label_x)
    plt.xticks(rotation=90)
    plt.bar(label_x, y1, align="edge", width=-0.35, label="tensorrt-llm")
    plt.bar(label_x, y2, align="edge", width=0.35, label="fastertransformer")
    
    plt.title("Execution time")
    plt.tight_layout()
    plt.savefig("./llama_compare.svg")
            

if __name__ == '__main__':
    main()
