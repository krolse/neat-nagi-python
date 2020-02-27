import pickle
import matplotlib.pyplot as plt

with open('../data/test_run.pkl', 'rb') as file:
    data = pickle.load(file)

for item in data:
    print(item)


average_fitnesses, max_fitnesses = data[0], data[1]
x = range(len(average_fitnesses))
fig = plt.figure()
plt.ylabel('fitness')
plt.xlabel('generation')
plt.plot(x, average_fitnesses, 'b')
plt.plot(x, max_fitnesses, 'r')
plt.show()
