from sklearn.metrics import accuracy_score
y_true = []
y_predict = []
with open('results.txt', 'r') as file:
    for line in file:
        y_predict.append(int(line.strip()))
with open('results_set_2.txt', 'r') as file:
    for line in file:
        y_true.append(int(line.strip()))

accuracy = accuracy_score(y_true, y_predict)
print(f"Accuracy: {accuracy}")