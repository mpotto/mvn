import matplotlib.pyplot as plt
import numpy as np


from mvn.data import generate_doubly_correlated_homoscedastic

plt.style.use("mvn.mplstyle")


def main():
    m, n, p = 5, 5, 5
    beta = np.arange(p)
    r = 10**3  # Replicates

    i, j = 2, 4
    ip, jp = 1, 3
    samples = np.zeros((r, 2))

    for k in range(r):
        Y, _, _, _ = generate_doubly_correlated_homoscedastic(
            m,
            n,
            beta,
            col_corr=0.5,
            row_corr=0.5,
            col_var=1,
            row_var=1,
            seed=0,
            generator=k,
        )
        samples[k, :] = Y[i, j], Y[ip, jp]

    print(np.corrcoef(samples, rowvar=False))


if __name__ == "__main__":
    main()
