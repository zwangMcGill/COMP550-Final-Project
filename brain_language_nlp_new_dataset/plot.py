import matplotlib.pyplot as plt
import numpy as np

file1 = open('./graph/xl_net_results.txt', 'r')
Lines = file1.readlines()
xticks = [1, 5, 10, 15, 20, 25, 30, 35]

fig, ax = plt.subplots()
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.rainbow(np.linspace(0, 1, 23))))
count = 0
# Strips the newline character
for line in Lines:
    count +=1
    if count % 2 == 0 and count!=2:
        line = line.strip().strip('][').split(', ')
        results = [float(i) for i in line]
        print(results)
        ax.set_xlabel('Sequence Length',fontsize=14)
        ax.set_ylabel('Accuracy',fontsize=14)
        ax.plot(xticks, results, label='layer' + str(int(count / 2)))
        ax.legend(fontsize=10)
        ax.set_xlim(0, 45)
plt.title('XLNet prediction accuracy on fMRI data')
plt.show()


file1 = open('./graph/results.txt', 'r')
Lines = file1.readlines()

fig, ax = plt.subplots()
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.rainbow(np.linspace(0, 1, 12))))

count = 0
# Strips the newline character
for line in Lines:
    count += 1
    if count % 2 == 0 and count != 16:
        line = line.strip().strip('][').split(', ')
        results = [float(i) for i in line]
        print(results)
        ax.set_xlabel('Sequence Length',fontsize=14)
        ax.set_ylabel('Accuracy',fontsize=14)
        ax.plot(xticks, results, label='layer' + str(int(count / 2)))
        ax.legend(fontsize=14)
        ax.set_xlim(0, 45)

plt.title('BERT prediction accuracy on fMRI data')
plt.show()


file1 = open('./graph/gpt_result.txt', 'r')
Lines = file1.readlines()


fig, ax = plt.subplots()
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.rainbow(np.linspace(0, 1, 12))))

count = 0
# Strips the newline character
for line in Lines:
    count +=1
    if count % 2 == 0 and count !=16:
        line = line.strip().strip('][').split(', ')
        results = [float(i) for i in line]
        print(results)
        ax.set_xlabel('Sequence Length',fontsize=14)
        ax.set_ylabel('Accuracy',fontsize=14)
        ax.plot(xticks, results, label= 'layer' + str(int(count/2)))
        ax.legend(fontsize=14)
        ax.set_xlim(0, 45)

plt.title('GPT2 prediction accuracy on fMRI data')
plt.show()

file1 = open('./graph/new_bert.txt', 'r')
Lines = file1.readlines()

xticks = [1,5,10,15,20]
fig, ax = plt.subplots()
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.rainbow(np.linspace(0, 1, 12))))
ax.set_xlabel('Sequence Length',fontsize=14)
ax.set_ylabel('Accuracy',fontsize=14)
count = 0
# Strips the newline character
for line in Lines:
    count +=1
    if count % 2 == 0 and count !=16:
        line = line.strip().strip('][').split(', ')
        results = [float(i) for i in line]
        print(results)
        ax.plot(xticks, results, label= 'layer' + str(12- int(count/2)))
        ax.legend(fontsize=14)
        ax.set_xlim(0, 25)

plt.title('Bert prediction accuracy on new fMRI data')
plt.show()

# f = open('./graph/bert_new_data.txt', 'r')
# Lines = f.readlines()
#
# layer = 0
# xticks = [1, 5, 10, 15, 20]
# fig, ax = plt.subplots()
# plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.rainbow(np.linspace(0, 1, 12))))
#
# for line in Lines:
#     line = line.strip().strip('][').split(', ')
#     acc = [float(i) for i in line]
#     ax.plot(xticks, acc, label='layer' + str(layer))
#     ax.legend()
#     ax.set_xlim(0, 25)
#     layer = layer + 1
#
# ax.set_xlabel('Context Length')
# ax.set_ylabel('Accuracy')
# plt.title('Bert Prediction Accuracy')
# plt.show()
