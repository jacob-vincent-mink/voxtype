//! Cohere GGUF inference through transcribe.cpp's `transcribe-cli`.
//!
//! This stays separate from the ONNX implementation because GGUF is a
//! transcribe.cpp model format, not an ONNX Runtime execution provider.

use crate::config::{CohereConfig, Config};
use crate::error::TranscribeError;
use crate::transcribe::Transcriber;
use std::io::Read;
use std::path::PathBuf;
use std::process::Command;

pub struct CohereGgufTranscriber {
    model: PathBuf,
    cli: PathBuf,
    backend: String,
    language: String,
    threads: Option<usize>,
}

impl CohereGgufTranscriber {
    pub fn new(config: &CohereConfig) -> Result<Self, TranscribeError> {
        let configured = PathBuf::from(&config.model);
        let model = if configured.exists() || configured.is_absolute() {
            configured
        } else {
            Config::models_dir().join(configured)
        };
        let mut file = std::fs::File::open(&model)
            .map_err(|e| TranscribeError::ModelNotFound(format!("{}: {e}", model.display())))?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic).map_err(|e| {
            TranscribeError::InitFailed(format!(
                "Cannot read GGUF header at {}: {e}",
                model.display()
            ))
        })?;
        if &magic != b"GGUF" {
            return Err(TranscribeError::InitFailed(format!(
                "{} is not a GGUF model",
                model.display()
            )));
        }
        let backend = config.gguf_backend.to_ascii_lowercase();
        if !matches!(
            backend.as_str(),
            "auto" | "cpu" | "cpu_accel" | "vulkan" | "metal" | "cuda" | "rocm"
        ) {
            return Err(TranscribeError::ConfigError(format!(
                "Invalid cohere.gguf_backend: {}",
                config.gguf_backend
            )));
        }
        let cli = config
            .gguf_cli_path
            .as_deref()
            .unwrap_or("transcribe-cli")
            .into();
        Ok(Self {
            model,
            cli,
            backend,
            language: config.language.clone(),
            threads: config.threads.filter(|&n| n > 0),
        })
    }
}

impl Transcriber for CohereGgufTranscriber {
    fn transcribe(&self, samples: &[f32]) -> Result<String, TranscribeError> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let wav = tempfile::Builder::new()
            .prefix("voxtype_cohere_")
            .suffix(".wav")
            .tempfile()
            .map_err(|e| TranscribeError::AudioFormat(e.to_string()))?;
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(wav.path(), spec)
            .map_err(|e| TranscribeError::AudioFormat(e.to_string()))?;
        for &sample in samples {
            writer
                .write_sample((sample.clamp(-1.0, 1.0) * 32767.0) as i16)
                .map_err(|e| TranscribeError::AudioFormat(e.to_string()))?;
        }
        writer
            .finalize()
            .map_err(|e| TranscribeError::AudioFormat(e.to_string()))?;

        let transcript = tempfile::Builder::new()
            .prefix("voxtype_cohere_")
            .suffix(".txt")
            .tempfile()
            .map_err(|e| TranscribeError::InferenceFailed(e.to_string()))?;
        let mut command = Command::new(&self.cli);
        command
            .arg("--model")
            .arg(&self.model)
            .arg("--backend")
            .arg(&self.backend)
            .arg("--language")
            .arg(&self.language)
            .arg("--timestamps")
            .arg("none")
            .arg("--quiet")
            .arg("--output")
            .arg(transcript.path());
        if let Some(threads) = self.threads {
            command.arg("--threads").arg(threads.to_string());
        }
        let output = command.arg(wav.path()).output().map_err(|e| {
            TranscribeError::InferenceFailed(format!(
                "Could not start {}: {e}. Install transcribe.cpp with Vulkan support or set cohere.gguf_cli_path",
                self.cli.display()
            ))
        })?;
        if !output.status.success() {
            return Err(TranscribeError::InferenceFailed(format!(
                "transcribe-cli failed ({}): {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        std::fs::read_to_string(transcript.path())
            .map(|s| s.trim().to_string())
            .map_err(|e| {
                TranscribeError::InferenceFailed(format!(
                    "Could not read transcribe-cli output: {e}"
                ))
            })
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn gguf_routes_to_vulkan_cli_and_reads_transcript() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("cohere-transcribe-03-2026-Q4_K_M.gguf");
        std::fs::write(&model, b"GGUFtest").unwrap();
        let cli = dir.path().join("transcribe-cli");
        std::fs::write(&cli, b"#!/bin/sh\nwhile [ $# -gt 0 ]; do\n  case \"$1\" in\n    --backend) [ \"$2\" = vulkan ] || exit 10; shift 2;;\n    --output) printf 'hello from GPU\\n' > \"$2\"; shift 2;;\n    *) shift;;\n  esac\ndone\n").unwrap();
        let mut permissions = std::fs::metadata(&cli).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&cli, permissions).unwrap();
        let config = CohereConfig {
            model: model.to_string_lossy().into_owned(),
            gguf_cli_path: Some(cli.to_string_lossy().into_owned()),
            gguf_backend: "vulkan".into(),
            ..CohereConfig::default()
        };
        let transcriber = CohereGgufTranscriber::new(&config).unwrap();
        assert_eq!(
            transcriber.transcribe(&[0.0; 160]).unwrap(),
            "hello from GPU"
        );
    }
}
