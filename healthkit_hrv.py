"""
HRV Feature Computation Module
===============================
Computes time-domain and frequency-domain HRV features from HeartRate data
within the HealthKit aggregation pipeline.

Integrates with HealthKitAggregator to compute HRV features per window.

Time-domain features:
- SDNN: Standard deviation of NN intervals (ms)
- RMSSD: Root mean square of successive differences (ms)
- pNN50: Percentage of successive intervals differing by >50ms
- pNN20: Percentage of successive intervals differing by >20ms
- HRV Triangular Index: Histogram-based measure

Frequency-domain features:
- VLF power (ms²): Very-low-frequency band (0.0033–0.04 Hz)
- LF power (ms², nu): Low-frequency band (0.04–0.15 Hz)
- HF power (ms², nu): High-frequency band (0.15–0.4 Hz)
- LF/HF ratio: Sympathovagal balance indicator
- Total power (ms²)

Note: Frequency analysis requires interpolation of irregularly-sampled
NN intervals to create evenly-sampled time series for FFT.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings

# Optional scipy imports for frequency-domain analysis
try:
    from scipy import signal, interpolate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available; frequency-domain HRV features will be disabled")

warnings.filterwarnings('ignore')


class HRVComputer:
    """Compute HRV features from HeartRate observations within windows."""
    
    def __init__(self,
                 min_samples: int = 10,
                 time_domain: bool = True,
                 freq_domain: bool = True,
                 freq_min_samples: int = 30,
                 interpolation_fs: float = 4.0):
        """
        Parameters
        ----------
        min_samples : int
            Minimum HR samples for time-domain features (default: 10)
        time_domain : bool
            Compute time-domain features (default: True)
        freq_domain : bool
            Compute frequency-domain features (default: True)
        freq_min_samples : int
            Minimum HR samples for frequency-domain features (default: 30)
        interpolation_fs : float
            Sampling rate for RR interpolation in Hz (default: 4.0)
        """
        self.min_samples = min_samples
        self.time_domain = time_domain
        self.freq_domain = freq_domain and SCIPY_AVAILABLE
        self.freq_min_samples = freq_min_samples
        self.interpolation_fs = interpolation_fs
        
        if freq_domain and not SCIPY_AVAILABLE:
            warnings.warn("scipy not available; frequency-domain features disabled")
            self.freq_domain = False
    
    def compute_window_hrv(self, 
                           hr_values: np.ndarray,
                           timestamps: np.ndarray) -> Dict[str, float]:
        """
        Compute HRV features for a single window.
        
        Parameters
        ----------
        hr_values : np.ndarray
            Heart rate values (bpm) in the window
        timestamps : np.ndarray
            Timestamps (datetime64 or seconds since epoch)
            
        Returns
        -------
        Dict with HRV features (keys match aggregation column naming)
        """
        results = {}
        
        if len(hr_values) < self.min_samples:
            return self._empty_results()
        
        # Convert HR to RR intervals (ms)
        rr_intervals = 60000.0 / hr_values
        
        # Clean RR intervals
        nn_intervals, valid_idx = self._clean_rr_intervals(rr_intervals)
        
        if len(nn_intervals) < self.min_samples:
            return self._empty_results()
        
        # Time-domain features
        if self.time_domain:
            td_features = self._compute_time_domain(
                nn_intervals=nn_intervals,
                hr_values=hr_values[valid_idx]
            )
            results.update(td_features)
        
        # Frequency-domain features
        if self.freq_domain and len(nn_intervals) >= self.freq_min_samples:
            # Get timestamps for valid intervals
            valid_timestamps = timestamps[valid_idx]
            timestamps_sec = self._to_seconds(valid_timestamps)
            
            fd_features = self._compute_freq_domain(
                nn_intervals=nn_intervals,
                timestamps_sec=timestamps_sec
            )
            results.update(fd_features)
        
        return results
    
    def _clean_rr_intervals(self, 
                           rr_intervals: np.ndarray,
                           min_rr: float = 300,
                           max_rr: float = 2000,
                           ectopic_threshold: float = 0.20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clean RR intervals by removing outliers and ectopic beats.
        
        Parameters
        ----------
        rr_intervals : np.ndarray
            RR intervals in milliseconds
        min_rr : float
            Minimum physiological RR interval (ms)
        max_rr : float
            Maximum physiological RR interval (ms)
        ectopic_threshold : float
            Threshold for ectopic beat detection (fraction of median)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (cleaned_rr_intervals, valid_indices)
        """
        if len(rr_intervals) == 0:
            return np.array([]), np.array([])
        
        indices = np.arange(len(rr_intervals))
        
        # Step 1: Physiological range filter
        mask = (rr_intervals >= min_rr) & (rr_intervals <= max_rr)
        rr = rr_intervals[mask]
        indices = indices[mask]
        
        if len(rr) < 2:
            return rr, indices
        
        # Step 2: Ectopic beat removal (MAD-based)
        median_rr = np.median(rr)
        lower = median_rr * (1 - ectopic_threshold)
        upper = median_rr * (1 + ectopic_threshold)
        mask2 = (rr >= lower) & (rr <= upper)
        rr = rr[mask2]
        indices = indices[mask2]
        
        return rr, indices
    
    def _compute_time_domain(self, 
                            nn_intervals: np.ndarray,
                            hr_values: np.ndarray) -> Dict[str, float]:
        """
        Compute time-domain HRV measures.
        
        Parameters
        ----------
        nn_intervals : np.ndarray
            Cleaned NN intervals (ms)
        hr_values : np.ndarray
            Corresponding HR values (bpm)
            
        Returns
        -------
        Dict with time-domain features
        """
        results = {}
        
        # Basic statistics
        results['nn_count'] = len(nn_intervals)
        results['mean_nn_ms'] = np.mean(nn_intervals)
        results['median_nn_ms'] = np.median(nn_intervals)
        results['mean_hr_bpm'] = np.mean(hr_values)
        results['hr_range_bpm'] = np.max(hr_values) - np.min(hr_values)
        
        # SDNN: Standard deviation of NN intervals
        if len(nn_intervals) >= 2:
            results['sdnn_ms'] = np.std(nn_intervals, ddof=1)
        else:
            results['sdnn_ms'] = np.nan
        
        # RMSSD: Root mean square of successive differences
        if len(nn_intervals) >= 2:
            diff = np.diff(nn_intervals)
            results['rmssd_ms'] = np.sqrt(np.mean(diff ** 2))
        else:
            results['rmssd_ms'] = np.nan
        
        # pNN50 and pNN20: Percentage of successive differences > threshold
        if len(nn_intervals) >= 2:
            diff_abs = np.abs(np.diff(nn_intervals))
            results['pnn50_pct'] = 100.0 * np.sum(diff_abs > 50) / len(diff_abs)
            results['pnn20_pct'] = 100.0 * np.sum(diff_abs > 20) / len(diff_abs)
        else:
            results['pnn50_pct'] = np.nan
            results['pnn20_pct'] = np.nan
        
        # HRV Triangular Index
        if len(nn_intervals) >= 10:
            results['hrv_tri_index'] = self._hrv_triangular_index(nn_intervals)
        else:
            results['hrv_tri_index'] = np.nan
        
        return results
    
    def _compute_freq_domain(self,
                            nn_intervals: np.ndarray,
                            timestamps_sec: np.ndarray) -> Dict[str, float]:
        """
        Compute frequency-domain HRV measures.
        
        Parameters
        ----------
        nn_intervals : np.ndarray
            Cleaned NN intervals (ms)
        timestamps_sec : np.ndarray
            Timestamps in seconds since first observation
            
        Returns
        -------
        Dict with frequency-domain features
        """
        results = {}
        
        # Interpolate to uniform sampling
        t_uniform, nn_interp = self._interpolate_nn(
            nn_intervals, 
            timestamps_sec, 
            self.interpolation_fs
        )
        
        if len(nn_interp) < 16:
            return self._empty_freq_results()
        
        # Calculate PSD using Welch's method
        freqs, psd = self._calculate_psd_welch(nn_interp, self.interpolation_fs)
        
        if len(freqs) == 0:
            return self._empty_freq_results()
        
        # Define frequency bands (Hz)
        bands = {
            'vlf': (0.0033, 0.04),
            'lf': (0.04, 0.15),
            'hf': (0.15, 0.4),
        }
        
        powers = {}
        peaks = {}
        for band_name, (f_low, f_high) in bands.items():
            power, peak = self._band_power(freqs, psd, f_low, f_high)
            powers[band_name] = power
            peaks[band_name] = peak
        
        # Store absolute powers
        results['vlf_power_ms2'] = powers['vlf']
        results['lf_power_ms2'] = powers['lf']
        results['lf_peak_hz'] = peaks['lf']
        results['hf_power_ms2'] = powers['hf']
        results['hf_peak_hz'] = peaks['hf']
        
        # Normalized units (LF and HF only)
        lf_hf_sum = powers['lf'] + powers['hf'] if not np.isnan(powers['lf']) and not np.isnan(powers['hf']) else 0
        
        if lf_hf_sum > 0:
            results['lf_power_nu'] = (powers['lf'] / lf_hf_sum) * 100
            results['hf_power_nu'] = (powers['hf'] / lf_hf_sum) * 100
        else:
            results['lf_power_nu'] = np.nan
            results['hf_power_nu'] = np.nan
        
        # LF/HF ratio
        if not np.isnan(powers['lf']) and not np.isnan(powers['hf']) and powers['hf'] > 0:
            results['lf_hf_ratio'] = powers['lf'] / powers['hf']
        else:
            results['lf_hf_ratio'] = np.nan
        
        # Total power
        total = sum(p for p in powers.values() if not np.isnan(p))
        results['total_power_ms2'] = total if total > 0 else np.nan
        
        return results
    
    def _interpolate_nn(self, 
                       nn_intervals: np.ndarray, 
                       timestamps_sec: np.ndarray, 
                       fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate NN intervals to uniform sampling rate.
        
        Parameters
        ----------
        nn_intervals : np.ndarray
            NN intervals (ms)
        timestamps_sec : np.ndarray
            Timestamps (seconds)
        fs : float
            Target sampling frequency (Hz)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (uniform_timestamps, interpolated_nn_intervals)
        """
        if len(nn_intervals) < 4:
            return np.array([]), np.array([])
        
        # Remove duplicate timestamps
        unique_mask = np.diff(timestamps_sec, prepend=-np.inf) > 0
        timestamps_sec = timestamps_sec[unique_mask]
        nn_intervals = nn_intervals[unique_mask]
        
        if len(nn_intervals) < 4:
            return np.array([]), np.array([])
        
        duration = timestamps_sec[-1] - timestamps_sec[0]
        if duration <= 0:
            return np.array([]), np.array([])
        
        n_samples = int(duration * fs)
        if n_samples < 4:
            return np.array([]), np.array([])
        
        t_uniform = np.linspace(timestamps_sec[0], timestamps_sec[-1], n_samples)
        
        try:
            # Try cubic spline interpolation
            f_interp = interpolate.interp1d(
                timestamps_sec, nn_intervals, 
                kind='cubic', fill_value='extrapolate'
            )
            nn_interp = f_interp(t_uniform)
        except:
            # Fall back to linear interpolation
            nn_interp = np.interp(t_uniform, timestamps_sec, nn_intervals)
        
        return t_uniform, nn_interp
    
    def _calculate_psd_welch(self, 
                            nn_interp: np.ndarray, 
                            fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate power spectral density using Welch's method.
        
        Parameters
        ----------
        nn_interp : np.ndarray
            Interpolated NN intervals
        fs : float
            Sampling frequency (Hz)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (frequencies, power_spectral_density)
        """
        if len(nn_interp) < 16:
            return np.array([]), np.array([])
        
        # Choose segment length
        nperseg = min(256, len(nn_interp) // 4)
        nperseg = max(nperseg, 16)
        noverlap = nperseg // 2
        
        try:
            freqs, psd = signal.welch(
                nn_interp, fs=fs,
                nperseg=nperseg, noverlap=noverlap,
                detrend='linear', scaling='density'
            )
        except:
            return np.array([]), np.array([])
        
        return freqs, psd
    
    def _band_power(self, 
                   freqs: np.ndarray, 
                   psd: np.ndarray, 
                   f_low: float, 
                   f_high: float) -> Tuple[float, float]:
        """
        Calculate power and peak frequency in a frequency band.
        
        Parameters
        ----------
        freqs : np.ndarray
            Frequency values (Hz)
        psd : np.ndarray
            Power spectral density
        f_low : float
            Lower frequency bound (Hz)
        f_high : float
            Upper frequency bound (Hz)
            
        Returns
        -------
        Tuple[float, float]
            (band_power_ms2, peak_frequency_hz)
        """
        if len(freqs) == 0:
            return np.nan, np.nan
        
        mask = (freqs >= f_low) & (freqs <= f_high)
        if not np.any(mask):
            return np.nan, np.nan
        
        band_freqs = freqs[mask]
        band_psd = psd[mask]
        
        # Compute power using trapezoidal integration
        if len(band_freqs) < 2:
            power = band_psd[0] * (f_high - f_low)
        else:
            power = np.trapz(band_psd, band_freqs)
        
        # Find peak frequency
        peak_idx = np.argmax(band_psd)
        peak_freq = band_freqs[peak_idx]
        
        return power, peak_freq
    
    def _hrv_triangular_index(self, 
                              nn_intervals: np.ndarray, 
                              bin_width: float = 7.8125) -> float:
        """
        Calculate HRV Triangular Index.
        
        Parameters
        ----------
        nn_intervals : np.ndarray
            NN intervals (ms)
        bin_width : float
            Histogram bin width (ms), default 7.8125 ms (128 Hz sampling)
            
        Returns
        -------
        float
            HRV Triangular Index
        """
        bins = np.arange(np.min(nn_intervals), np.max(nn_intervals) + bin_width, bin_width)
        if len(bins) < 2:
            return np.nan
        hist, _ = np.histogram(nn_intervals, bins=bins)
        max_height = np.max(hist)
        return len(nn_intervals) / max_height if max_height > 0 else np.nan
    
    def _to_seconds(self, timestamps) -> np.ndarray:
        """
        Convert timestamps to seconds from first timestamp.
        
        Parameters
        ----------
        timestamps : np.ndarray
            Timestamps (datetime64 or similar)
            
        Returns
        -------
        np.ndarray
            Timestamps in seconds since first timestamp
        """
        if len(timestamps) == 0:
            return np.array([])
        ts = pd.to_datetime(timestamps)
        t0 = ts.min()
        return np.array([(t - t0).total_seconds() for t in ts])
    
    def _empty_results(self) -> Dict[str, float]:
        """Return empty result dict for time-domain."""
        results = {
            'nn_count': 0,
            'mean_nn_ms': np.nan,
            'median_nn_ms': np.nan,
            'mean_hr_bpm': np.nan,
            'hr_range_bpm': np.nan,
            'sdnn_ms': np.nan,
            'rmssd_ms': np.nan,
            'pnn50_pct': np.nan,
            'pnn20_pct': np.nan,
            'hrv_tri_index': np.nan,
        }
        if self.freq_domain:
            results.update(self._empty_freq_results())
        return results
    
    def _empty_freq_results(self) -> Dict[str, float]:
        """Return empty result dict for frequency-domain."""
        return {
            'vlf_power_ms2': np.nan,
            'lf_power_ms2': np.nan,
            'lf_peak_hz': np.nan,
            'lf_power_nu': np.nan,
            'hf_power_ms2': np.nan,
            'hf_peak_hz': np.nan,
            'hf_power_nu': np.nan,
            'lf_hf_ratio': np.nan,
            'total_power_ms2': np.nan,
        }


def compute_hrv_features_for_windows(
    windows_df: pd.DataFrame,
    hr_intermediate: pd.DataFrame,
    hrv_computer: Optional[HRVComputer] = None
) -> pd.DataFrame:
    """
    Compute HRV features for each window from HeartRate intermediate data.
    
    Parameters
    ----------
    windows_df : pd.DataFrame
        Window definitions with columns: user_id, window_id, window_start, window_end
    hr_intermediate : pd.DataFrame
        HeartRate intermediate data from aggregator
        Columns: user_id, window_id, window_start, window_end, value, utc_start (or point_ts_utc)
    hrv_computer : HRVComputer, optional
        Configured HRV computer (creates default if None)
        
    Returns
    -------
    pd.DataFrame
        HRV features with one row per window, wide format
        Columns prefixed with 'HRV__'
    """
    if hrv_computer is None:
        hrv_computer = HRVComputer()
    
    if hr_intermediate is None or len(hr_intermediate) == 0:
        return pd.DataFrame()
    
    # Determine timestamp column
    ts_col = 'point_ts_utc' if 'point_ts_utc' in hr_intermediate.columns else 'utc_start'
    
    # Group by window and compute HRV
    results = []
    grouped = hr_intermediate.groupby(['user_id', 'window_id', 'window_start', 'window_end'])
    
    for (user_id, win_id, win_start, win_end), group in grouped:
        hr_values = group['value'].values
        timestamps = group[ts_col].values
        
        # Compute HRV features
        hrv_features = hrv_computer.compute_window_hrv(hr_values, timestamps)
        
        # Add identifiers
        hrv_features['user_id'] = user_id
        hrv_features['window_id'] = win_id
        hrv_features['window_start'] = win_start
        hrv_features['window_end'] = win_end
        
        results.append(hrv_features)
    
    if len(results) == 0:
        return pd.DataFrame()
    
    hrv_df = pd.DataFrame(results)
    
    # Prefix columns with 'HRV__' to match aggregation naming convention
    rename_cols = {
        col: f'HRV__{col}' 
        for col in hrv_df.columns 
        if col not in ['user_id', 'window_id', 'window_start', 'window_end']
    }
    hrv_df = hrv_df.rename(columns=rename_cols)
    
    return hrv_df
