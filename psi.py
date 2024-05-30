import numpy as np
import warnings

def cast_to_np(array_like):
    """
    Convert input data to a numpy array.
    
    Parameters:
    array_like (array-like): Input data to be converted.
    
    Returns:
    np.ndarray: Numpy array converted from input data.
    """
    return np.array(array_like)

def calculate_psi(actual_data: np.ndarray, new_data: np.ndarray, bins: int = 10) -> float:
    """
    Calculate the Population Stability Index (PSI) to measure the distributional
    shift between the actual and new data distributions.

    Parameters:
    actual_data (np.ndarray): Array of actual observed data.
    new_data (np.ndarray): Array of new observed data.
    bins (int, optional): Number of bins to use for the histograms. Default is 10.

    Returns:
    float: The PSI value indicating the distribution shift.

    Interpretation of PSI values:
    - Less than 0.1: No significant change
    - 0.1 to 0.2: Moderate change
    - Greater than 0.2: Significant change
    """
    actual_data, new_data = cast_to_np(actual_data), cast_to_np(new_data)
    
    # Check if the data is numerical or categorical
    if not (np.issubdtype(actual_data.dtype, np.integer) or np.issubdtype(actual_data.dtype, np.floating)):
        combined_data = np.concatenate([actual_data, new_data])
        unique_categories, encoded_data = np.unique(combined_data, return_inverse=True)

        actual_data = encoded_data[:len(actual_data)]
        new_data = encoded_data[len(actual_data):]
        bins = len(unique_categories)
        
        warnings.warn("Categorical features detected; ignoring provided bins value and using the number of unique categories instead.")

    # Define the range for the histograms based on the min and max values of both datasets
    data_range = (min(actual_data.min(), new_data.min()), max(actual_data.max(), new_data.max()))

    # Calculate the fractions of actual and new data in each bin
    actual_hist, _ = np.histogram(actual_data, bins=bins, range=data_range)
    new_hist, _ = np.histogram(new_data, bins=bins, range=data_range)

    # Normalize the histogram counts to get proportions
    actual_fractions = actual_hist / len(actual_data)
    new_fractions = new_hist / len(new_data)

    # Avoid division by zero by replacing zero values with a small number
    actual_fractions = np.where(actual_fractions == 0, 0.0001, actual_fractions)
    new_fractions = np.where(new_fractions == 0, 0.0001, new_fractions)

    # Calculate the PSI for each bin and sum them up
    psi_values = (actual_fractions - new_fractions) * np.log(actual_fractions / new_fractions)
    psi_value = np.sum(psi_values)

    return psi_value
