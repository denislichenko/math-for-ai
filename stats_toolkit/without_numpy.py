import math


def get_mean(arr):
    """
    Calculate the arithmetic mean of a dataset.

    Formula:
        mean = sum(x) / n

    Meaning:
        The central value around which data is distributed.
    """
    return sum(arr) / len(arr)


def get_deviations(arr, mean):
    """
    Calculate deviations from the mean.

    Formula:
        deviation_i = x_i - mean

    Meaning:
        Shows how far each value is from the center.
    """
    return [x - mean for x in arr]


def get_squared_deviations(deviations):
    """
    Square deviations to remove negative values.

    Formula:
        (x_i - mean)^2

    Meaning:
        Penalizes large deviations more strongly.
    """
    return [d ** 2 for d in deviations]


def get_variance(arr):
    """
    Calculate sample variance (unbiased estimator).

    Formula:
        variance = sum((x - mean)^2) / (n - 1)

    Meaning:
        Measures how spread out the data is.
    """
    mean = get_mean(arr)
    deviations = get_deviations(arr, mean)
    squared = get_squared_deviations(deviations)
    return sum(squared) / (len(arr) - 1)


def get_standard_deviation(arr):
    """
    Calculate standard deviation.

    Formula:
        std = sqrt(variance)

    Meaning:
        Typical distance from the mean (same units as data).
    """
    return math.sqrt(get_variance(arr))


def get_standard_error(arr):
    """
    Calculate standard error of the mean.

    Formula:
        SE = std / sqrt(n)

    Meaning:
        How much the sample mean would vary across samples.
    """
    std = get_standard_deviation(arr)
    return std / math.sqrt(len(arr))


def get_confidence_interval(mean, standard_error, z=1.96):
    """
    Calculate confidence interval for the mean.

    Formula:
        mean ± z * SE

    Meaning:
        Range where true mean lies with given confidence.
    """
    return (
        mean - z * standard_error,
        mean + z * standard_error
    )


def normalize(arr):
    """
    Z-score normalization.

    Formula:
        z = (x - mean) / std

    Meaning:
        Centers data at 0 with unit variance.
        Critical for ML optimization.
    """
    mean = get_mean(arr)
    std = get_standard_deviation(arr)
    return [(x - mean) / std for x in arr]


def print_detailed_stats(arr):
    """
    Print step-by-step statistical calculations.
    """
    print("DATA:", arr)
    print("-" * 50)

    mean = get_mean(arr)
    print(f"Mean (μ): {mean}")

    deviations = get_deviations(arr, mean)
    print(f"Deviations (x - μ): {deviations}")

    squared = get_squared_deviations(deviations)
    print(f"Squared deviations: {squared}")
    print(f"Sum of squared deviations: {sum(squared)}")

    variance = get_variance(arr)
    print(f"Variance (σ²): {variance}")

    std = get_standard_deviation(arr)
    print(f"Standard deviation (σ): {std}")

    se = get_standard_error(arr)
    print(f"Standard error (SE): {se}")

    ci = get_confidence_interval(mean, se)
    print(f"95% confidence interval for mean: {ci}")

    normalized = normalize(arr)
    print(f"Z-score normalized data: {normalized}")

    print("-" * 50)
    print("INTERPRETATION:")
    print(f"- Typical value deviates from mean by ~{std:.2f}")
    print(f"- Mean estimate noise is ~{se:.2f}")
    print(f"- More data → narrower confidence interval")
    print()


# Example usage
data = [2, 4, 6, 8, 5, 3, 2, 1, 5, 6]
print_detailed_stats(data)
