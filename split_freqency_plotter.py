import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
columns_to_import = ['y_r', 'y_l']
sampling_rate = 1000
segment_count = 10
likelihood_threshold = 0.95
def filter_indices(csv_file, col, likelihood_col):
    """Filter rows based on likelihood threshold."""
    raw_data = pd.read_csv(csv_file)
    lh = raw_data[likelihood_col].values
    indices = np.where(lh > likelihood_threshold)[0]
    filtered_y = raw_data[col].values[indices]
    filtered_coords = raw_data['coords'].values[indices]
    return filtered_y, filtered_coords

def process_fft_segment(values, sampling_rate):
    """Compute dominant frequency using FFT."""
    values_detrended = values - np.mean(values)
    fft_result = np.fft.fft(values_detrended)
    fft_freq = np.fft.fftfreq(len(values_detrended), d=1 / sampling_rate)
    
    fft_magnitude = np.abs(fft_result)
    positive_freqs = fft_freq[:len(fft_freq)//2]
    positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]

    dominant_idx = int(np.argmax(positive_magnitude[1:])) + 1
    dominant_freq = positive_freqs[dominant_idx]

    # Check frequency range
    if 10 < dominant_freq < 50:
        return dominant_freq
    else:
        # fallback to second highest magnitude freq
        sorted_indices = np.argsort(positive_magnitude)
        return positive_freqs[sorted_indices[-2]]

def process_file(file_path, column, likelihood_col):
    """Segment and compute dominant frequencies."""
    df = pd.read_csv(file_path)
    if column not in df.columns:
        print(f"Column '{column}' not found in {file_path}. Skipping.")
        return None
    
    filtered_y, filtered_coords = filter_indices(file_path, column, likelihood_col)
    if len(filtered_y) == 0:
        print(f"No data points passed likelihood filter in {file_path} for {column}.")
        return None

    splits = np.array_split(filtered_y, segment_count)
    coord_splits = np.array_split(filtered_coords, segment_count)

    frequencies = [process_fft_segment(seg, sampling_rate) if len(seg) > 1 else np.nan for seg in splits]
    return frequencies, splits, coord_splits, filtered_y, filtered_coords

def plot_frequencies(file_path, column_name, likelihood_col, frequencies, splits, coord_splits, filtered_y, filtered_coords, save_dir):
    """Plot and save frequency data."""
    file_base = os.path.basename(file_path)
    moth_info = os.path.basename(os.path.dirname(file_path))

    filename = f"{file_base[:-40]}_{column_name}_segment_{segment_count}.tif"
    save_path = os.path.join(save_dir, filename)

    fig, axes = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'wspace': 0.15}, constrained_layout=True, squeeze=False)

    # Left plot: filtered signal with segment boundaries
    axes[0, 0].plot(filtered_coords, filtered_y, label='Filtered Signal', color='blue') # type: ignore
    for segment in coord_splits:
        axes[0, 0].axvline(segment[0], color='red', alpha=0.5)
        axes[0, 0].axvline(segment[-1], color='red', alpha=0.5)
    axes[0, 0].set_title('Wingbeat Signal with Segments')
    axes[0, 0].set_xlabel('Coordinates')
    axes[0, 0].set_ylabel('y-coordinate of the marked point on the wing')
    axes[0, 0].legend()

    # Put info text as figure suptitle or in left plot legend area (optional)
    info_text = (
        f"Moth info: {moth_info}\n"
        f"Trial info: {file_base[:-45]}\n"
        f"Coordinate wing: {column_name}\n"
        f"Segment size: {len(splits[0])}\n"
        f"likelihood threshold: {likelihood_threshold}\n"
    )
    axes[0, 0].text(0.8, -0.17, info_text, fontsize=12, ha='center', va='top', transform=axes[0, 0].transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

    # Right plot: frequencies per segment
    axes[0, 1].plot(np.arange(len(frequencies)), frequencies, marker='o', linestyle='-')
    axes[0, 1].set_title('Dominant Frequency per Segment')
    axes[0, 1].set_xlabel('Segment Number')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    axes[0, 1].set_xticks(np.arange(len(frequencies)))
    axes[0, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    current_filepath = os.path.abspath(__file__)
    base_dir = os.path.dirname(current_filepath)

    save_folder = os.path.join(base_dir, 'split_freq')
    os.makedirs(save_folder, exist_ok=True)

    for filename in os.listdir(base_dir):
        if not filename.endswith('.csv'):
            continue

        file_path = os.path.join(base_dir, filename)

        for column in columns_to_import:
            likelihood_col = 'likelihood_r' if column == 'y_r' else 'likelihood_l'
            result = process_file(file_path, column, likelihood_col)
            if result:
                frequencies, splits, coord_splits, filtered_y, filtered_coords = result
                plot_frequencies(file_path, column, likelihood_col, frequencies, splits, coord_splits, filtered_y, filtered_coords, save_folder)
