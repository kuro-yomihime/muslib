use std::fs::File;
use std::path::PathBuf;

use hound::{WavSpec, WavWriter};
use symphonia::core::audio::{AudioBuffer, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::conv::ConvertibleSample;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Loader provides a facility for audio input.
///
/// That means you can
/// load files by their filenames,
/// mix down tracks from that file down to mono,
/// and finally inspect separate audio frames.
///
/// Essentia has a few layers of classes to provide this functionality:
///  AudioLoader -> MonoLoader -> EasyLoader.
/// In muslib this has been unified with a single Loader.
/// Essentia is using ffmpeg for the AudioLoader, we however are using Symphonia.
pub struct Loader<T> {
    file_path: PathBuf,
    gain: Option<f64>,
    channel: Option<usize>,
    track: Option<usize>,
    sample_rate: Option<u32>,
    data: Vec<T>,
}

impl<T: ConvertibleSample> Default for Loader<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ConvertibleSample> Loader<T> {
    /// creates a new Loader instance with empty or default values
    pub fn new() -> Self {
        Loader {
            file_path: PathBuf::from(""),
            gain: None,        // will default to neutral
            channel: None,     // None will mix down all the channels to mono
            track: None,       // defaults to the first track
            sample_rate: None, // will be discovered on .load()
            data: Vec::new(),
        }
    }

    /// set path to a file that this Loader will read
    pub fn file(&mut self, file_path: PathBuf) -> &mut Self {
        self.file_path = file_path;
        self
    }

    /// set gain value that will be used while mixing all the channels
    pub fn gain(&mut self, gain: f64) -> &mut Self {
        self.gain = Some(gain);
        self
    }

    /// pick a single channel to read from
    pub fn channel(&mut self, channel: usize) -> &mut Self {
        self.channel = Some(channel);
        self
    }

    /// read all channels and mix them down to mono
    pub fn mono(&mut self) -> &mut Self {
        self.channel = None;
        self
    }

    /// execute the Loader to load and mix the data
    pub fn load(&mut self) -> Result<&Self, Error> {
        let file = File::open(&self.file_path).expect("Could not open file path for this Loader.");

        let mut hint = Hint::new();
        if let Some(ext) = self.file_path.extension() {
            if let Some(ext_s) = ext.to_str() {
                hint.with_extension(ext_s);
            }
        }

        let mss_opts = MediaSourceStreamOptions::default();
        let mss = MediaSourceStream::new(Box::new(file), mss_opts);

        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();
        let probe = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .expect("Could not recognize and load this file format.");

        let mut format = probe.format;
        let track = self
            .track
            .and_then(|t| format.tracks().get(t))
            .or_else(|| {
                format
                    .tracks()
                    .iter()
                    .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            })
            .expect("Could not find any supported audio tracks.");
        let track_id = track.id;

        let decode_opts = DecoderOptions::default();
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &decode_opts)
            .expect("The codec is unsupported.");

        // decode
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                // finished reading the file
                Err(Error::ResetRequired) => {
                    return Ok(self);
                }
                Err(Error::DecodeError(_)) | Err(Error::IoError(_)) => {
                    return Ok(self);
                }
                // real errors
                Err(err) => {
                    return Err(err);
                }
            };

            if packet.track_id() != track_id {
                continue;
            }

            while !format.metadata().is_latest() {
                format.metadata().pop();
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    let spec = *decoded.spec();
                    if let Some(rate) = self.sample_rate {
                        if rate != spec.rate {
                            // sample rate mismatch error between reads, stream changed
                            return Err(Error::ResetRequired);
                        }
                    } else {
                        self.sample_rate = Some(spec.rate);
                    }

                    let mut data = AudioBuffer::<T>::new(decoded.capacity() as u64, spec);
                    decoded.convert(&mut data);

                    let len = decoded.frames(); // n of samples in each channel
                    let channels = data.spec().channels.count();

                    let mut buf = Vec::<T>::with_capacity(len);
                    let channel = data.chan(self.channel.unwrap_or(0));

                    for ch in channel.iter().take(len) {
                        buf.push(self.apply_gain(*ch));
                    }

                    if self.channel.is_none() {
                        // mixing down to mono
                        for ch in 1..channels {
                            let channel = data.chan(ch);
                            for i in 0..len {
                                buf[i] = buf[i] + self.apply_gain(channel[i]);
                            }
                        }
                    }

                    self.data.append(&mut buf);
                }
                Err(Error::ResetRequired) => {
                    // stream changed, so we finished reading the file
                    return Ok(self);
                }
                Err(Error::DecodeError(_)) | Err(Error::IoError(_)) => {
                    // the packet can be discarded
                    continue;
                }
                Err(err) => return Err(err),
            }
        }
    }

    fn apply_gain(&self, x: T) -> T {
        // TODO
        x
    }

    /// read the loaded pcm data as a vector
    pub fn data(&self) -> Vec<T> {
        self.data.clone()
    }

    /// read the loaded sample rate
    pub fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }
}

/// Writer provides a facility for audio output.
///
/// Essentia is using ffmpeg for the AudioWriter,
/// we however are using hound and for now only support output to simple 16-bit WAV files.
pub struct Writer {
    file_path: PathBuf,
    spec: WavSpec,
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer {
    /// creates a new Writer instance with empty or default values
    pub fn new() -> Self {
        Writer {
            file_path: PathBuf::from(""),
            spec: WavSpec {
                channels: 1,
                sample_rate: 44100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        }
    }

    /// set path to a file that this Writer will use
    pub fn file(&mut self, file_path: PathBuf) -> &mut Self {
        self.file_path = file_path;
        self
    }

    /// set sample rate that will be used with this Writer
    pub fn sample_rate(&mut self, sample_rate: u32) -> &mut Self {
        self.spec.sample_rate = sample_rate;
        self
    }

    /// execute the Writer to store data in a file
    pub fn write(&self, data: &[u16]) -> Result<(), ()> {
        let mut writer = WavWriter::create(&self.file_path, self.spec)
            .expect("Failed to create a file for the Writer.");

        for t in data.iter() {
            let t = (*t ^ 0x8000) as i16; // hack for the sign conversion
            writer
                .write_sample(t)
                .expect("Failed to write an output sample.");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use hound::{SampleFormat, WavSpec, WavWriter};

    use super::{Loader, Writer};

    /// create a unique temporary path for a test
    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("muslib_rs_test_{}_{name}", std::process::id()))
    }

    fn cleanup(path: &PathBuf) {
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn writer_loader_roundtrip_u16() {
        let path = temp_path("roundtrip_u16");
        let data: [u16; 6] = [0, 16384, 32767, 32768, 49152, 65535];

        Writer::new()
            .sample_rate(8000)
            .file(path.clone())
            .write(&data)
            .unwrap();

        let mut loader = Loader::<u16>::new();
        loader.file(path.clone()).load().unwrap();

        assert_eq!(loader.data(), data, "unsigned pcm survives the roundtrip");
        assert_eq!(loader.sample_rate(), Some(8000));

        cleanup(&path);
    }

    #[test]
    fn loader_reads_signed_samples() {
        let path = temp_path("signed");
        let data: [u16; 6] = [0, 16384, 32767, 32768, 49152, 65535];

        Writer::new().file(path.clone()).write(&data).unwrap();

        let mut loader = Loader::<i16>::new();
        loader.file(path.clone()).load().unwrap();

        let expected: Vec<i16> = data.iter().map(|&v| (v ^ 0x8000) as i16).collect();
        assert_eq!(loader.data(), expected, "signed conversion is an xor");

        cleanup(&path);
    }

    #[test]
    fn loader_reads_normalized_float_samples() {
        let path = temp_path("float");
        let data: [u16; 4] = [0, 16384, 32768, 65535];

        Writer::new().file(path.clone()).write(&data).unwrap();

        let mut loader = Loader::<f32>::new();
        loader.file(path.clone()).load().unwrap();

        for (loaded, &raw) in loader.data().iter().zip(data.iter()) {
            let expected = ((raw ^ 0x8000) as i16) as f32 / 32768.0;
            assert!(
                (loaded - expected).abs() < 1e-6,
                "loaded {} expected {}",
                loaded,
                expected
            );
        }

        cleanup(&path);
    }

    #[test]
    fn loader_mixes_down_stereo_to_mono() {
        let path = temp_path("stereo_mono");

        // build a stereo wav with hound directly: ch0 = [100, 200, -300],
        // ch1 = [-50, 50, 100]
        let spec = WavSpec {
            channels: 2,
            sample_rate: 8000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(&path, spec).unwrap();
        for s in [(100i16, -50i16), (200, 50), (-300, 100)] {
            writer.write_sample(s.0).unwrap();
            writer.write_sample(s.1).unwrap();
        }
        writer.finalize().unwrap();

        let mut mono = Loader::<i16>::new();
        mono.file(path.clone()).load().unwrap();
        assert_eq!(mono.data(), vec![50, 250, -200], "sum of both channels");

        let mut first = Loader::<i16>::new();
        first.file(path.clone()).channel(0).load().unwrap();
        assert_eq!(first.data(), vec![100, 200, -300], "channel 0 only");

        let mut second = Loader::<i16>::new();
        second.file(path.clone()).channel(1).load().unwrap();
        assert_eq!(second.data(), vec![-50, 50, 100], "channel 1 only");

        cleanup(&path);
    }

    #[test]
    #[should_panic(expected = "Could not open file path")]
    fn loader_panics_on_missing_file() {
        let mut loader = Loader::<f32>::new();
        loader.file(temp_path("does_not_exist")).load().unwrap();
    }

    #[test]
    #[should_panic(expected = "Failed to create a file")]
    fn writer_panics_on_unwritable_path() {
        let mut path = std::env::temp_dir();
        path.push("muslib_rs_no_such_dir");
        path.push("out.wav");

        Writer::new().file(path).write(&[0]).unwrap();
    }
}
