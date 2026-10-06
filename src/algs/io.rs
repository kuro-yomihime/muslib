use pyo3::{pyclass, pymethods};

use super::Algorithm;
use crate::mixer::{Loader, Writer};

#[pyclass(get_all)]
pub struct MonoLoader {
    /// Input: str -- path to a file that will be loaded
    #[pyo3(set)]
    pub file: String,
    /// Output: Optional[list[int]] -- raw 16-bit pcm values of loaded data
    pub pcm_data: Option<Vec<u16>>,
    /// Output: int -- sample rate
    pub sample_rate: usize,
}

#[pymethods]
impl MonoLoader {
    #[new]
    fn pynew() -> Self {
        MonoLoader {
            file: "".into(),
            pcm_data: None,
            sample_rate: 0,
        }
    }

    /// Compute the Algorithm
    ///
    /// Inputs:
    ///   - file: str
    ///
    /// Outputs:
    ///   - pcm_data: list[int]
    ///   - sample_rate: int
    ///
    /// See data descriptors for more details.
    #[pyo3(name = "compute", signature = (file=None))]
    fn pycompute(&mut self, file: Option<String>) -> (Vec<u16>, usize) {
        if let Some(arg) = file {
            self.file = arg
        }

        self.compute();

        (self.pcm_data.as_ref().unwrap().clone(), self.sample_rate)
    }

    fn __call__(&mut self) {
        self.compute()
    }
}

impl Algorithm for MonoLoader {
    fn new() -> Self {
        Self::pynew()
    }

    fn compute(&mut self) {
        let mut loader = Loader::<u16>::new();
        loader
            .file(self.file.clone().into())
            .load()
            .expect("Load failed");
        self.pcm_data = Some(loader.data());
        self.sample_rate = loader.sample_rate().unwrap() as usize;
    }
}

#[pyclass(get_all)]
pub struct MonoWriter {
    /// Input: str -- path to a file that will be written
    #[pyo3(set)]
    pub file: String,
    /// Input: list[int] -- raw 16-bit pcm values of data to be written
    pub pcm_data: Vec<u16>,
    /// Param: int -- sample rate
    #[pyo3(set)]
    pub sample_rate: usize,
}

#[pymethods]
impl MonoWriter {
    #[new]
    #[pyo3(signature = (
        sample_rate=44100,
    ))]
    fn pynew(sample_rate: usize) -> Self {
        MonoWriter {
            file: "".into(),
            pcm_data: Vec::new(),
            sample_rate,
        }
    }

    /// Compute the Algorithm
    ///
    /// Inputs:
    ///   - file: str
    ///   - pcm_data: list[int]
    ///
    /// See data descriptors for more details.
    #[pyo3(name = "compute", signature = (file=None, pcm_data=None))]
    fn pycompute(&mut self, file: Option<String>, pcm_data: Option<Vec<u16>>) {
        if let Some(arg) = file {
            self.file = arg
        }
        if let Some(arg) = pcm_data {
            self.pcm_data = arg
        }

        self.compute();
    }

    fn __call__(&mut self) {
        self.compute()
    }
}

impl Algorithm for MonoWriter {
    fn new() -> Self {
        Self::pynew(44100)
    }

    fn compute(&mut self) {
        let mut writer = Writer::new();
        writer
            .sample_rate(self.sample_rate as u32)
            .file(self.file.clone().into())
            .write(&self.pcm_data)
            .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::algs::Algorithm;
    use super::{MonoLoader, MonoWriter};

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("muslib_io_test_{}_{name}", std::process::id()))
    }

    #[test]
    fn mono_writer_mono_loader_roundtrip() {
        let path = temp_path("roundtrip");
        let data: [u16; 5] = [0, 10000, 32767, 50000, 65535];

        let mut writer = MonoWriter::new();
        writer.sample_rate = 8000;
        writer.file = path.to_str().unwrap().into();
        writer.pcm_data = data.into();
        writer.compute();

        let mut loader = MonoLoader::new();
        loader.file = path.to_str().unwrap().into();
        loader.compute();

        assert_eq!(loader.pcm_data.unwrap(), data);
        assert_eq!(loader.sample_rate, 8000);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    #[should_panic(expected = "called `Option::unwrap()`")] // no audio in the file
    fn mono_loader_empty_file_has_no_sample_rate() {
        let mut writer = MonoWriter::new();
        writer.sample_rate = 8000;
        writer.file = temp_path("empty").to_str().unwrap().into();
        writer.compute();

        let mut loader = MonoLoader::new();
        loader.file = writer.file.clone();
        loader.compute();

        let _ = std::fs::remove_file(writer.file);
    }

    #[test]
    fn mono_writer_defaults() {
        let w = MonoWriter::new();
        assert_eq!(w.sample_rate, 44100);
        assert!(w.pcm_data.is_empty());
        assert!(w.file.is_empty());
    }
}
