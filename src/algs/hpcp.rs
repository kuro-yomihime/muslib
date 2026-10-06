use pyo3::{pyclass, pymethods};

use super::Algorithm;

#[pyclass(get_all, set_all)]
pub struct HPCP {
    /// Input: list[float] -- frequencies of the spectral peaks
    pub frequencies: Vec<f64>,
    /// Input: list[float] -- magnitudes of spectral peaks
    pub magnitudes: Vec<f64>,

    /// Output: Optional[list[float]] -- resulting harmonic pitch class profile
    pub hpcp_data: Option<Vec<f64>>,

    /// Param: int -- the size of the output HPCP, one of {12, 24, 36} (default: 12)
    pub size: usize,
    /// Param: float -- sampling rate of the audio signal in Hz (default: 44100)
    pub sample_rate: f64,
    /// Param: float -- the reference frequency for semitone index calculation (default: 440)
    pub reference_frequency: f64,
    /// Param: float -- split frequency for low and high bands (default: 500)
    pub band_split_frequency: f64,
    /// Param: float -- maximum frequency that contributes to the HPCP in Hz (default: 5000)
    pub max_frequency: f64,
    /// Param: float -- minimum frequency that contributes to the HPCP in Hz (default: 40)
    pub min_frequency: f64,
    /// Param: int -- number of additional harmonics for frequency contribution (default: 0)
    pub harmonics: usize,
    /// Param: str -- whether to use a squared cosine weighting funcion for determining frequency contribution (default: true)
    pub weighting: bool,
    /// Param: float -- size in semitones of the window used for weighting
    pub weighting_window_size: f64,
    /// Param: bool -- whether to normalize output vectors (default: true)
    pub normalized: bool,
    /// Param: bool -- whether to apply nonlinear post-processin on output vectors (default: false)
    pub nonlinear_post: bool,

    harmonic_peaks: Vec<(f64, f64)>,
}

#[pymethods]
impl HPCP {
    #[new]
    #[pyo3(signature = (
        size=12,
        sample_rate=44100.0,
        reference_frequency=440.0,
        band_split_frequency=500.0,
        max_frequency=5000.0,
        min_frequency=40.0,
        harmonics=0,
        weighting=true,
        weighting_window_size=1.0,
        normalized=true,
        nonlinear_post=false,
    ))]
    fn pynew(
        size: usize,
        sample_rate: f64,
        reference_frequency: f64,
        band_split_frequency: f64,
        max_frequency: f64,
        min_frequency: f64,
        harmonics: usize,
        weighting: bool,
        weighting_window_size: f64,
        normalized: bool,
        nonlinear_post: bool,
    ) -> Self {
        HPCP {
            frequencies: Vec::new(),
            magnitudes: Vec::new(),

            hpcp_data: None,

            size,
            sample_rate,
            reference_frequency,
            band_split_frequency,
            max_frequency,
            min_frequency,
            harmonics,
            weighting,
            weighting_window_size,
            normalized,
            nonlinear_post,

            harmonic_peaks: Vec::new(),
        }
    }

    /// Compute the Algorithm
    ///
    /// Inputs:
    ///   - frequencies: list[float]
    ///   - magnitudes: list[float]
    ///
    /// Outputs:
    ///   - hpcp_data: list[float]
    ///
    /// See data descriptors for more details.
    #[pyo3(name = "compute", signature = (frequencies=None, magnitudes=None))]
    fn pycompute(
        &mut self,
        frequencies: Option<Vec<f64>>,
        magnitudes: Option<Vec<f64>>,
    ) -> Vec<f64> {
        if let Some(arg) = frequencies {
            self.frequencies = arg
        }
        if let Some(arg) = magnitudes {
            self.magnitudes = arg
        }

        self.compute();

        self.hpcp_data.as_ref().unwrap().clone()
    }

    fn __call__(&mut self) {
        self.compute()
    }
}

impl Algorithm for HPCP {
    fn new() -> Self {
        Self::pynew(
            12, 44100.0, 440.0, 500.0, 5000.0, 40.0, 0, true, 1.0, true, false,
        )
    }

    fn compute(&mut self) {
        self.adjust_input();
        self.init_harmonic_peaks();

        let mut output_low = vec![0.0; self.size];
        let mut output_high = vec![0.0; self.size];

        for i in 0..self.frequencies.len() {
            let freq = self.frequencies[i];
            let mag = self.magnitudes[i];

            if freq >= self.min_frequency && freq <= self.max_frequency {
                if freq < self.band_split_frequency {
                    self.add_contribution(freq, mag, &mut output_low);
                } else {
                    self.add_contribution(freq, mag, &mut output_high);
                }
            }
        }

        // normalize each band
        if self.normalized {
            Self::normalize(&mut output_low);
            Self::normalize(&mut output_high);
        }

        for i in 0..self.size {
            output_high[i] += output_low[i];
        }

        // normalize the sum again
        if self.normalized {
            Self::normalize(&mut output_high);

            if self.nonlinear_post {
                for i in 0..self.size {
                    output_high[i] = (output_high[i] * std::f64::consts::PI / 2.0).sin().powi(2);
                    if output_high[i] < 0.6 {
                        output_high[i] *= (output_high[i] / 0.6).powi(2);
                    }
                }
            }
        }

        // Output
        self.hpcp_data = Some(output_high);
    }
}

impl HPCP {
    fn maxf(a: f64, b: f64) -> f64 {
        if a > b {
            a
        } else {
            b
        }
    }

    fn maxvf(target: &Vec<f64>) -> f64 {
        let mut m = 0.0;
        for x in target {
            let x = *x;
            if x > m {
                m = x;
            }
        }
        m
    }

    fn normalize(target: &mut Vec<f64>) {
        let m = Self::maxvf(target);
        if m == 0.0 {
            return;
        }

        for i in 0..target.len() {
            target[i] /= m;
        }
    }

    fn adjust_input(&mut self) {
        // adjust the size
        if self.size <= 12 {
            self.size = 12
        } else if self.size <= 24 {
            self.size = 24
        } else {
            self.size = 36
        }

        // adjust the window size
        self.weighting_window_size =
            Self::maxf(self.weighting_window_size, 12.0 / (self.size as f64));

        // adjust size of input vectors to shorter of the two
        let f_l = self.frequencies.len();
        let m_l = self.magnitudes.len();
        if f_l > m_l {
            self.frequencies.truncate(m_l)
        } else if m_l > f_l {
            self.magnitudes.truncate(f_l)
        }
    }

    fn init_harmonic_peaks(&mut self) {
        self.harmonic_peaks.clear();

        // fundamental frequency
        self.harmonic_peaks.push((0.0, 1.0));

        // semitones
        for i in 0..self.harmonics {
            let mut semitone = 12.0 * (1.0 + i as f64).log2();
            let octweight = Self::maxf(1.0, (semitone / 12.0) * 0.5);
            let precision = 1e-5;
            while semitone >= 12.0 - precision {
                semitone -= 12.0
            }

            let mut r = 0;
            for j in 1..self.harmonic_peaks.len() {
                let harmonic = self.harmonic_peaks[j];
                if harmonic.0 > semitone - precision && harmonic.0 < semitone + precision {
                    r = j
                }
            }

            if r == 0 {
                self.harmonic_peaks.push((semitone, (1.0 / octweight)));
            } else {
                self.harmonic_peaks[r].1 += 1.0 / octweight;
            }
        }
    }

    fn add_contribution(&self, freq: f64, mag: f64, target: &mut [f64]) {
        for harmonic in &self.harmonic_peaks {
            let f = freq * (2.0_f64).powf(-harmonic.0 / 12.0);
            let w = harmonic.1;

            let size = self.size as f64;
            let bin_f = size * (freq / self.reference_frequency).log2();

            if self.weighting {
                // add contributions with weight

                let resolution = size / 12.0;
                let left = (bin_f - resolution * self.weighting_window_size / 2.0).ceil() as i64;
                let right =
                    1 + (bin_f + resolution * self.weighting_window_size / 2.0).floor() as i64;

                // skip invalid
                if right < left {
                    continue;
                }

                // apply weight to all bins in the window
                for i in left..right {
                    let distance =
                        (bin_f - i as f64).abs() / (resolution * self.weighting_window_size);
                    let weight = (std::f64::consts::PI * distance).cos().powi(2);
                    let bin = ((i + self.size as i64) as usize) % self.size;
                    target[bin] += weight * mag.powi(2) * w.powi(2);
                }
            } else {
                // add contribution without weight

                // skip invalid
                if f <= 0.0 {
                    continue;
                }

                let bin = ((size + bin_f.round()) % size) as usize;
                target[bin] += mag.powi(2) * w.powi(2);
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::{Algorithm, HPCP};

    #[test]
    fn hpcp() {
        let input = [
            (vec![880.0], vec![1.0]),
            (vec![440.0, 660.0], vec![0.5, 1.0]),
        ];

        let result = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ];

        let mut hpcp = HPCP::new();
        hpcp.band_split_frequency = 0.0;

        for i in 0..input.len() {
            hpcp.frequencies.clone_from(&input[i].0);
            hpcp.magnitudes.clone_from(&input[i].1);
            hpcp.compute();
            let r: Vec<f64> = hpcp
                .hpcp_data
                .as_ref()
                .unwrap()
                .clone()
                // round for the poor with precision to the 2nd decimal place
                .iter()
                .map(|x| (*x * 100.0).round() / 100.0)
                .collect();
            assert_eq!(r, result[i], "test {}", i);
        }
    }

    #[test]
    fn hpcp_adjusts_size_to_allowed_values() {
        // size snaps to the closest allowed value: 12, 24 or 36
        for (size, expected) in [(5usize, 12usize), (13, 24), (30, 36)] {
            let mut hpcp = HPCP::new();
            hpcp.size = size;
            hpcp.compute();

            let r = hpcp.hpcp_data.expect("hpcp_data computed");
            assert_eq!(r.len(), expected, "size {}", size);
            assert!(r.iter().all(|&x| x == 0.0), "empty input gives zeros");
        }
    }

    #[test]
    fn hpcp_ignores_out_of_band_frequencies() {
        // frequencies outside <min_frequency; max_frequency> do not contribute
        let mut hpcp = HPCP::new();
        hpcp.band_split_frequency = 0.0;
        hpcp.frequencies = vec![20.0, 880.0, 10000.0];
        hpcp.magnitudes = vec![1.0, 1.0, 1.0];
        hpcp.compute();

        let r = hpcp.hpcp_data.expect("hpcp_data computed");
        assert_eq!(r[0], 1.0, "only 880 Hz contributes, at bin 0");
        assert!(r[1..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn hpcp_truncates_input_to_shorter_vector() {
        // extra frequencies beyond the magnitudes length are discarded
        let mut hpcp = HPCP::new();
        hpcp.band_split_frequency = 0.0;
        hpcp.frequencies = vec![440.0, 880.0, 1320.0];
        hpcp.magnitudes = vec![1.0];
        hpcp.compute();

        let r = hpcp.hpcp_data.expect("hpcp_data computed");
        assert_eq!(r[0], 1.0, "only the first peak was used");
        assert_eq!(r[7], 0.0, "1320 Hz would land at bin 7 if not truncated");
    }

    #[test]
    fn hpcp_unnormalized_keeps_raw_contributions() {
        // without normalization the output holds the raw magnitude squares
        let mut hpcp = HPCP::new();
        hpcp.band_split_frequency = 0.0;
        hpcp.normalized = false;
        hpcp.frequencies = vec![440.0];
        hpcp.magnitudes = vec![0.5];
        hpcp.compute();

        let r = hpcp.hpcp_data.expect("hpcp_data computed");
        assert_eq!(r[0], 0.25, "0.5 squared");
        assert!(r[1..].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn hpcp_normalizes_bands_separately() {
        // with the default band split each band is normalized on its own
        let mut hpcp = HPCP::new();
        hpcp.frequencies = vec![440.0, 660.0]; // 440 Hz -> low band, 660 Hz -> high band
        hpcp.magnitudes = vec![0.5, 1.0];
        hpcp.compute();

        let r = hpcp.hpcp_data.expect("hpcp_data computed");
        assert_eq!(r[0], 1.0, "low band peak");
        assert_eq!(r[7], 1.0, "high band peak");
        assert!(r[1..7].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn hpcp_weighting_applies_cosine_window() {
        let mut weighted = HPCP::new();
        weighted.band_split_frequency = 0.0;
        weighted.normalized = false;
        weighted.frequencies = vec![660.0]; // bin_f = 12 * log2(660/440) ~= 7.0196
        weighted.magnitudes = vec![1.0];
        weighted.compute();

        let mut unweighted = weighted.clone_fields();
        unweighted.weighting = false;
        unweighted.compute();

        let w = weighted.hpcp_data.expect("weighted computed");
        let u = unweighted.hpcp_data.expect("unweighted computed");

        assert_eq!(u[7], 1.0, "unweighted peak lands in bin 7 exactly");
        assert!(
            (w[7] - (std::f64::consts::PI * (12.0 * (660.0f64 / 440.0).log2() - 7.0))
                .cos()
                .powi(2))
            .abs()
                < 1e-9,
            "weighted peak is scaled by the squared cosine window"
        );
    }

    #[test]
    fn hpcp_harmonics_increase_fundamental_weight() {
        // harmonics contribute at the peak bin with weight w^2 each; for
        // harmonics = 3 the harmonic peaks carry weights 1, 2 and 1, so the
        // unnormalized contribution grows from 1 to 1 + 4 + 1
        for (harmonics, expected) in [(0usize, 1.0), (3, 6.0)] {
            let mut hpcp = HPCP::new();
            hpcp.band_split_frequency = 0.0;
            hpcp.normalized = false;
            hpcp.harmonics = harmonics;
            hpcp.frequencies = vec![440.0];
            hpcp.magnitudes = vec![1.0];
            hpcp.compute();

            let r = hpcp.hpcp_data.expect("hpcp_data computed");
            assert_eq!(r[0], expected, "harmonics {}", harmonics);
        }
    }

    #[test]
    fn hpcp_nonlinear_post_suppresses_weak_bins() {
        // nonlinear post-processing squashes values well below the maximum
        let mut linear = HPCP::new();
        linear.band_split_frequency = 0.0;
        linear.frequencies = vec![440.0, 660.0];
        linear.magnitudes = vec![0.5, 1.0];
        linear.compute();
        let l = linear.hpcp_data.clone().expect("linear computed");

        let mut nonlinear = linear.clone_fields();
        nonlinear.nonlinear_post = true;
        nonlinear.compute();
        let n = nonlinear.hpcp_data.expect("nonlinear computed");

        assert_eq!(n[7], 1.0, "maximum bin is preserved");
        assert!(
            n[0] < l[0] * 0.1,
            "weak bin {}: suppressed below 10% of linear {}",
            n[0],
            l[0]
        );
    }

    impl HPCP {
        /// copy the fields of this HPCP into a new instance for comparisons
        fn clone_fields(&self) -> HPCP {
            HPCP {
                frequencies: self.frequencies.clone(),
                magnitudes: self.magnitudes.clone(),
                hpcp_data: None,
                size: self.size,
                sample_rate: self.sample_rate,
                reference_frequency: self.reference_frequency,
                band_split_frequency: self.band_split_frequency,
                max_frequency: self.max_frequency,
                min_frequency: self.min_frequency,
                harmonics: self.harmonics,
                weighting: self.weighting,
                weighting_window_size: self.weighting_window_size,
                normalized: self.normalized,
                nonlinear_post: self.nonlinear_post,
                harmonic_peaks: Vec::new(),
            }
        }
    }
}
