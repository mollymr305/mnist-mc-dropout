"""Plot training/validation error."""
import cPickle as pickle
import gzip
import matplotlib.pyplot as plt
import seaborn as sns


# Unpickle training info
filename = './output/mnist_mc_dropout_info.pkl.gz'
f = gzip.open(filename, 'rb')
data = pickle.load(f)
f.close()

# Get accuracy
TA = data['training accuracy']
VA = data['validation accuracy']

# Compute error as %
TE = [(1. - ta) * 100 for ta in TA]
VE = [(1. - va) * 100 for va in VA]

# Plot training and validation errors
plt.switch_backend('Agg')
plt.plot(VE, c='r', lw=1.5)
plt.plot(TE, c='g', lw=1.5)
plt.legend(['validation', 'training'], fontsize=20)
plt.ylim(-1, 10)
plt.xlabel('epoch', size=20)
plt.ylabel('error (%)', size=20)
plt.tight_layout()
plt.savefig('./output/errors.eps', format='eps', dpi=100)
plt.savefig('./output/errors.jpg', format='jpg', dpi=100)
plt.close()