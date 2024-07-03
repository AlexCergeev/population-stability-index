import numpy as np
import warnings

def cast_to_np(array_like):
    """
    Converts input data to a numpy array.

    Parameters:
    array_like (array-like): Input data to be converted.

    Returns:
    np.ndarray: Converted numpy array.
    """
    return np.array(array_like)

def cast_categories(actual_data, new_data):
    combined_data = np.concatenate([actual_data, new_data])
    unique_categories, encoded_data = np.unique(combined_data, return_inverse=True)

    actual_data = encoded_data[:len(actual_data)]
    new_data = encoded_data[len(actual_data):]
    bins = len(unique_categories)
    
    warnings.warn("Categorical features detected; bins set to the number of unique categories.")
    return actual_data, new_data, bins

def calculate_psi(actual_data: np.ndarray, new_data: np.ndarray, bins: int = 10) -> float:
    """
    Calculates the Population Stability Index (PSI) to measure shifts between data distributions.

    Parameters:
    actual_data (np.ndarray): Array of actual observed data.
    new_data (np.ndarray): Array of new observed data.
    bins (int, optional): Number of bins for histograms. Default is 10.

    Returns:
    float: PSI value indicating the distribution shift.

    Interpretation of PSI values:
    - Less than 0.1: No significant change
    - 0.1 to 0.2: Moderate change
    - Greater than 0.2: Significant change
    """
    actual_data, new_data = cast_to_np(actual_data), cast_to_np(new_data)
    
    if not (np.issubdtype(actual_data.dtype, np.integer) or np.issubdtype(actual_data.dtype, np.floating)):
        actual_data, new_data, bins = cast_categories(actual_data, new_data)
        
    data_range = (min(actual_data.min(), new_data.min()), max(actual_data.max(), new_data.max()))

    actual_hist, _ = np.histogram(actual_data, bins=bins, range=data_range)
    new_hist, _ = np.histogram(new_data, bins=bins, range=data_range)

    actual_fractions = actual_hist / len(actual_data)
    new_fractions = new_hist / len(new_data)

    actual_fractions = np.where(actual_fractions == 0, 0.0001, actual_fractions)
    new_fractions = np.where(new_fractions == 0, 0.0001, new_fractions)

    psi_values = (actual_fractions - new_fractions) * np.log(actual_fractions / new_fractions)
    psi_value = np.sum(psi_values)

    return psi_value
