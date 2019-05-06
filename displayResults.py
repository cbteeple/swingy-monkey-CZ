import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

plt.close('all')
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'sans-serif'

# Load the training history, calulate stats, then plot
hist = np.load('hist_train.npy')
print('\n'+"TRAINING:_____________")
print("AVERAGE SCORE: %0.3f"%(np.mean(hist)))
print("MAX SCORE: %0.3f"%(np.max(hist)))


plt.figure()
plt.plot(hist,'o')
plt.title('Training')
plt.xlabel("Epoch")
plt.ylabel("Game Score")
plt.savefig('results_train.png',dpi = 300)
plt.show()



# Load the testing history, calulate stats, then plot
hist = np.load('hist_test.npy')
print('\n'+"FINAL TEST:_____________")
print("AVERAGE SCORE: %0.3f"%(np.mean(hist)))
print("MAX SCORE: %0.3f"%(np.max(hist)))


plt.figure()
plt.plot(hist,'o')
plt.title('Testing')
plt.xlabel("Epoch")
plt.ylabel("Game Score")
plt.savefig('results_test.png',dpi = 300)
plt.show()