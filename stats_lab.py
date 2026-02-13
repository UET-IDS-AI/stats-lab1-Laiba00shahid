import numpy as np
import matplotlib.pyplot as plt


# =========================
# Q1 — Histograms
# =========================

def normal_histogram(n):
    """
    Generate n samples from Normal(0,1),
    plot histogram with 10 bins,
    and return the generated data.
    """
    data = np.random.normal(loc=0, scale=1, size=n)
    
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Normal(0,1) Distribution")
    plt.show()
    
    return data


def uniform_histogram(n):
    """
    Generate n samples from Uniform(0,10),
    plot histogram with 10 bins,
    and return the generated data.
    """
    data = np.random.uniform(low=0, high=10, size=n)
    
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Uniform(0,10) Distribution")
    plt.show()
    
    return data


def bernoulli_histogram(n):
    """
    Generate n samples from Bernoulli(0.5),
    plot histogram with 10 bins,
    and return the generated data.
    """
    data = np.random.binomial(n=1, p=0.5, size=n)
    
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Bernoulli(0.5) Distribution")
    plt.show()
    
    return data


# =========================
# Q2 — Sample Mean & Variance
# =========================

def sample_mean(data):
    """
    Compute sample mean.
    """
    return np.sum(data) / len(data)


def sample_variance(data):
    """
    Compute sample variance using (n-1) in denominator.
    """
    n = len(data)
    mean = sample_mean(data)
    return np.sum((data - mean) ** 2) / (n - 1)


# =========================
# Q3 — Order Statistics
# =========================

def order_statistics(data):
    """
    Return (min, max, median, q1, q3).
    For [5,1,3,2,4], this returns:
    (1, 5, 3, 2, 4)
    """
    sorted_data = np.sort(data)
    
    minimum = sorted_data[0]
    maximum = sorted_data[-1]
    median = np.median(sorted_data)
    
    # Q1 = 25th percentile
    q1 = np.percentile(sorted_data, 25)
    
    # Q3 = 75th percentile
    q3 = np.percentile(sorted_data, 75)
    
    return (minimum, maximum, median, q1, q3)


# =========================
# Q4 — Sample Covariance
# =========================

def sample_covariance(x, y):
    """
    Compute sample covariance using (n-1) denominator.
    """
    n = len(x)
    mean_x = sample_mean(x)
    mean_y = sample_mean(y)
    
    return np.sum((x - mean_x) * (y - mean_y)) / (n - 1)


# =========================
# Q5 — Covariance Matrix (2x2)
# =========================

def covariance_matrix(x, y):
    """
    Return 2x2 covariance matrix:
    
    [ Var(x)      Cov(x,y) ]
    [ Cov(x,y)    Var(y)   ]
    """
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)
    
    return np.array([[var_x, cov_xy],
                     [cov_xy, var_y]])

# =========================
# OUTPUT 
# =========================

# Q1 — Generate histograms
data_normal = normal_histogram(100)
data_uniform = uniform_histogram(100)
data_bernoulli = bernoulli_histogram(100)

# Q2 — Mean & Variance
print("Normal Mean:", sample_mean(data_normal))
print("Normal Variance:", sample_variance(data_normal))

# Q3 — Order Statistics
test_data = np.array([5, 1, 3, 2, 4])
print("Order Statistics:", order_statistics(test_data))

# Q4 — Sample Covariance
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
print("Sample Covariance:", sample_covariance(x, y))

# Q5 — Covariance Matrix
print("Covariance Matrix:\n", covariance_matrix(x, y))
