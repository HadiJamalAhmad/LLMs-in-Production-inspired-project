import numpy as np

def get_dict(data):
    words = sorted(list(set(data)))
    word2Ind = {w:i for i,w in enumerate(words)}
    Ind2word = {i:w for w,i in word2Ind.items()}
    return word2Ind, Ind2word


def get_batches(data, word2Ind, V, C, batch_size):
    for i in range(0, len(data) - C*2, batch_size):
        x_batch = []
        y_batch = []
        for j in range(i, min(i+batch_size, len(data)-C*2)):
            context = data[j:j+C] + data[j+C+1:j+2*C+1]
            target = data[j+C]

            x = np.mean([one_hot(word2Ind[w], V) for w in context], axis=0)
            y = one_hot(word2Ind[target], V)

            x_batch.append(x)
            y_batch.append(y)

        yield np.array(x_batch).T, np.array(y_batch).T


def one_hot(idx, V):
    vec = np.zeros(V)
    vec[idx] = 1
    return vec


def compute_pca(X, n_components=2):
    X = X - np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, idx]
    return np.dot(X, eig_vecs[:, :n_components])