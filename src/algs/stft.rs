use pyo3::{pyclass, pymethods};
use symphonia::core::dsp::complex::Complex;
use symphonia::core::dsp::fft::Fft;

use super::Algorithm;

#[pyclass(get_all)]
pub struct FFT {
    /// Input: list[float] -- audio input frame, max len 65535
    #[pyo3(set)]
    pub frame: Vec<f64>,
    /// Output: list[tuple[float, float]] -- fft data
    pub fft_data: Vec<(f32, f32)>,
}

#[pymethods]
impl FFT {
    #[new]
    fn pynew() -> Self {
        FFT {
            frame: Vec::new(),
            fft_data: Vec::new(),
        }
    }

    /// Compute the Algorithm
    ///
    /// Inputs:
    ///   - frame: list[float]
    ///
    /// Outputs:
    ///   - fft_data: list[tuple[float, float]]
    ///
    /// See data descriptors for more details.
    #[pyo3(name = "compute", signature = (frame = None))]
    fn pycompute(&mut self, frame: Option<Vec<f64>>) -> Vec<(f32, f32)> {
        if let Some(arg) = frame {
            self.frame = arg
        }

        self.compute();

        self.fft_data.clone()
    }
}

impl Algorithm for FFT {
    fn new() -> Self {
        Self::pynew()
    }

    fn compute(&mut self) {
        // contruct a buffer of complex numbers
        let mut buf: [Complex; Fft::MAX_SIZE] = [Complex { re: 0.0, im: 0.0 }; Fft::MAX_SIZE];
        let n = std::cmp::min(self.frame.len(), Fft::MAX_SIZE);
        let buf = &mut buf[0..n];
        for i in 0..n {
            buf[i].re = self.frame[i] as f32;
        }

        let fft = Fft::new(n);
        fft.fft_inplace(buf);

        // convert and store the buffer as output
        self.fft_data = Vec::with_capacity(n);
        for x in buf.iter().take(n) {
            self.fft_data.push((x.re, x.im))
        }
    }
}

#[pyclass(get_all)]
pub struct IFFT {
    /// Input: list[tuple[float, float]] -- fft data, max len 65535
    #[pyo3(set)]
    pub fft_data: Vec<(f64, f64)>,
    /// Output: list[float] -- the IFFT of the input frame
    pub frame: Vec<f32>,
}

#[pymethods]
impl IFFT {
    #[new]
    fn pynew() -> Self {
        IFFT {
            fft_data: Vec::new(),
            frame: Vec::new(),
        }
    }

    /// Compute the Algorithm
    ///
    /// Inputs:
    ///   - fft_data: list[tuple[float, float]]
    ///
    /// Outputs:
    ///   - frame: list[float]
    ///
    /// See data descriptors for more details.
    #[pyo3(name = "compute", signature = (fft_data = None))]
    fn pycompute(&mut self, fft_data: Option<Vec<(f64, f64)>>) -> Vec<f32> {
        if let Some(arg) = fft_data {
            self.fft_data = arg
        }

        self.compute();

        self.frame.clone()
    }
}

impl Algorithm for IFFT {
    fn new() -> Self {
        Self::pynew()
    }

    fn compute(&mut self) {
        // contruct a buffer of complex numbers
        let mut buf: [Complex; Fft::MAX_SIZE] = [Complex { re: 0.0, im: 0.0 }; Fft::MAX_SIZE];
        let n = std::cmp::min(self.fft_data.len(), Fft::MAX_SIZE);
        let buf = &mut buf[0..n];
        for i in 0..n {
            buf[i].re = self.fft_data[i].0 as f32;
            buf[i].im = self.fft_data[i].1 as f32;
        }

        let fft = Fft::new(n);
        fft.ifft_inplace(buf);

        // convert and store the buffer as output
        self.frame = Vec::with_capacity(n);
        for x in buf.iter().take(n) {
            self.frame.push(x.re)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Algorithm, FFT, IFFT};

    const EPS: f32 = 1e-3;

    fn assert_close(a: f32, b: f32, what: &str) {
        assert!(
            (a - b).abs() < EPS,
            "{}: got {}, expected {}",
            what,
            a,
            b
        );
    }

    #[test]
    fn fft_of_constant_is_dc_bin() {
        // a constant frame of n samples transforms to a single DC bin of n * value
        let mut fft = FFT::new();
        fft.frame = vec![0.5; 16];
        fft.compute();

        assert_eq!(fft.fft_data.len(), 16);
        for (k, &(re, im)) in fft.fft_data.iter().enumerate() {
            if k == 0 {
                assert_close(re, 8.0, "DC bin");
            } else {
                assert_close(re, 0.0, &format!("bin {} re", k));
            }
            assert_close(im, 0.0, &format!("bin {} im", k));
        }
    }

    #[test]
    fn fft_of_impulse_is_flat() {
        // Kronecker delta transforms to a flat spectrum of unit magnitude
        let mut fft = FFT::new();
        fft.frame = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        fft.compute();

        assert_eq!(fft.fft_data.len(), 8);
        for (k, &(re, im)) in fft.fft_data.iter().enumerate() {
            assert_close(re, 1.0, &format!("bin {} re", k));
            assert_close(im, 0.0, &format!("bin {} im", k));
        }
    }

    #[test]
    fn fft_peaks_at_signal_and_mirror_bin() {
        // a real cosine at bin 1 shows energy in bin 1 and its mirror bin n-1
        let n = 32;
        let mut fft = FFT::new();
        fft.frame = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * (i as f64) / (n as f64)).cos())
            .collect();
        fft.compute();

        for (k, &(re, _)) in fft.fft_data.iter().enumerate() {
            let expected = if k == 1 || k == n - 1 { n as f32 / 2.0 } else { 0.0 };
            assert_close(re, expected, &format!("bin {} re", k));
        }
    }

    #[test]
    fn fft_recomputes_without_accumulating() {
        // computing on a new frame must overwrite the previous output
        let mut fft = FFT::new();
        fft.frame = vec![1.0; 8];
        fft.compute();
        assert_close(fft.fft_data[0].0, 8.0, "first DC bin");

        fft.frame = vec![0.0; 8];
        fft.compute();
        assert_close(fft.fft_data[0].0, 0.0, "second DC bin");
    }

    // NOTE: truncation of inputs longer than Fft::MAX_SIZE (1 << 16) is NOT
    // testable: symphonia 0.5.3 truncates the permutation table index to u16,
    // so Fft::new(1 << 16) is broken upstream and every smaller size is not a
    // power of two. The largest working transform size is 1 << 15.

    #[test]
    fn ifft_of_dc_is_constant() {
        // DC-only spectrum of amplitude n inverts to a constant frame of 1.0
        let mut ifft = IFFT::new();
        ifft.fft_data = vec![(32.0, 0.0)]
            .into_iter()
            .chain(std::iter::repeat((0.0, 0.0)).take(31))
            .collect();
        ifft.compute();

        assert_eq!(ifft.frame.len(), 32);
        for (i, &x) in ifft.frame.iter().enumerate() {
            assert_close(x, 1.0, &format!("sample {}", i));
        }
    }

    #[test]
    fn ifft_of_flat_spectrum_is_impulse() {
        // flat unit spectrum inverts to a unit impulse at sample 0
        let mut ifft = IFFT::new();
        ifft.fft_data = vec![(1.0, 0.0); 32];
        ifft.compute();

        for (i, &x) in ifft.frame.iter().enumerate() {
            let expected = if i == 0 { 1.0 } else { 0.0 };
            assert_close(x, expected, &format!("sample {}", i));
        }
    }

    #[test]
    fn ifft_recomputes_without_accumulating() {
        let mut ifft = IFFT::new();
        ifft.fft_data = vec![(1.0, 0.0); 32];
        ifft.compute();
        assert_close(ifft.frame[0], 1.0, "first sample");

        ifft.fft_data = vec![(0.0, 0.0); 32];
        ifft.compute();
        assert_close(ifft.frame[0], 0.0, "second sample");
    }

    #[test]
    fn ifft_inverts_fft() {
        // FFT followed by IFFT recovers the original frame
        let frame: Vec<f64> = (0..64u64)
            .map(|i| {
                0.5 * (2.0 * std::f64::consts::PI * 3.0 * i as f64 / 64.0).sin()
                    + 0.25 * (2.0 * std::f64::consts::PI * 7.0 * i as f64 / 64.0).sin()
            })
            .collect();

        let mut fft = FFT::new();
        fft.frame = frame.clone();
        fft.compute();

        let mut ifft = IFFT::new();
        ifft.fft_data = fft.fft_data.iter().map(|&(re, im)| (re as f64, im as f64)).collect();
        ifft.compute();

        assert_eq!(ifft.frame.len(), frame.len());
        for (i, (&x, &original)) in ifft.frame.iter().zip(frame.iter()).enumerate() {
            assert!(
                (x - original as f32).abs() < 1e-4,
                "sample {}: got {}, expected {}",
                i,
                x,
                original
            );
        }
    }
}
