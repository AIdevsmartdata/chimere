//! # Weight Loader — Chimère Engine
//!
//! Loads safetensors model weights and remaps HuggingFace naming conventions
//! to Chimère's internal naming. Supports mmap for zero-copy loading.
//!
//! Also provides `Qwen35WeightLoader` for loading Qwen3.5 GGUF weights
//! with typed tensor access for GDN, attention, FFN, and MTP tensors.

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

use crate::config::{ChimereModelConfig, Qwen35Config};
use crate::gguf_loader::GgufFile;

/// Weight loader that wraps a VarBuilder with model-specific remapping.
pub struct WeightLoader<'a> {
    vb: VarBuilder<'a>,
    config: ChimereModelConfig,
}

impl<'a> WeightLoader<'a> {
    /// Load safetensors from a model directory (auto-discovers shards).
    ///
    /// The directory should contain either:
    /// - `model.safetensors` (single file)
    /// - `model-00001-of-*.safetensors` (sharded)
    pub fn from_dir(
        model_dir: impl AsRef<Path>,
        config: ChimereModelConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<WeightLoader<'static>> {
        let dir = model_dir.as_ref();

        // Discover safetensors files
        let mut safetensor_files: Vec<std::path::PathBuf> = Vec::new();
        let single = dir.join("model.safetensors");
        if single.exists() {
            safetensor_files.push(single);
        } else {
            // Look for sharded files
            let entries = std::fs::read_dir(dir)
                .map_err(|e| candle_core::Error::Msg(format!("Cannot read dir: {}", e)))?;
            for entry in entries {
                let entry = entry
                    .map_err(|e| candle_core::Error::Msg(format!("Dir entry error: {}", e)))?;
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("model-") && name.ends_with(".safetensors") {
                    safetensor_files.push(entry.path());
                }
            }
            safetensor_files.sort();
        }

        if safetensor_files.is_empty() {
            return Err(candle_core::Error::Msg(
                format!("No safetensors files found in {:?}", dir),
            ));
        }

        // Load via mmap (zero-copy)
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)?
        };

        Ok(WeightLoader { vb, config })
    }

    /// Get the underlying VarBuilder for a specific layer's attention weights.
    /// Maps: `model.layers.{i}.self_attn.*` → `*`
    pub fn layer_attn_vb(&self, layer_idx: usize) -> VarBuilder<'_> {
        self.vb.pp(format!("model.layers.{}.self_attn", layer_idx))
    }

    /// Get VarBuilder for a specific layer's MLP weights.
    /// Maps: `model.layers.{i}.mlp.*` → `*`
    pub fn layer_mlp_vb(&self, layer_idx: usize) -> VarBuilder<'_> {
        self.vb.pp(format!("model.layers.{}.mlp", layer_idx))
    }

    /// Get the input_layernorm weight for a layer.
    pub fn input_layernorm_weight(&self, layer_idx: usize) -> Result<Tensor> {
        self.vb.pp(format!("model.layers.{}.input_layernorm", layer_idx))
            .get_with_hints(&[self.config.hidden_size], "weight", Default::default())
    }

    /// Get the post_attention_layernorm weight for a layer.
    pub fn post_attention_layernorm_weight(&self, layer_idx: usize) -> Result<Tensor> {
        self.vb.pp(format!("model.layers.{}.post_attention_layernorm", layer_idx))
            .get_with_hints(&[self.config.hidden_size], "weight", Default::default())
    }

    /// Get the embedding weight: `model.embed_tokens.weight` → [vocab_size, hidden_size]
    pub fn embed_tokens_weight(&self) -> Result<Tensor> {
        self.vb.pp("model.embed_tokens")
            .get_with_hints(
                &[self.config.vocab_size, self.config.hidden_size],
                "weight",
                Default::default(),
            )
    }

    /// Get the final RMSNorm weight: `model.norm.weight` → [hidden_size]
    pub fn final_norm_weight(&self) -> Result<Tensor> {
        self.vb.pp("model.norm")
            .get_with_hints(&[self.config.hidden_size], "weight", Default::default())
    }

    /// Get the lm_head weight: `lm_head.weight` → [vocab_size, hidden_size]
    pub fn lm_head_weight(&self) -> Result<Tensor> {
        self.vb.pp("lm_head")
            .get_with_hints(
                &[self.config.vocab_size, self.config.hidden_size],
                "weight",
                Default::default(),
            )
    }

    /// Get the model config.
    pub fn config(&self) -> &ChimereModelConfig {
        &self.config
    }
}

/// Expand KV head weights from `[num_kv_heads * head_dim, hidden_size]` to
/// `[num_q_heads * head_dim, hidden_size]` by repeating each KV head's rows.
///
/// Used when mapping MHA KV projections to DeltaNet (which expects full num_heads).
pub fn expand_kv_weight(
    weight: &Tensor,
    num_kv_heads: usize,
    num_q_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if num_kv_heads == num_q_heads {
        return Ok(weight.clone());
    }

    let group_size = num_q_heads / num_kv_heads;
    // weight: [num_kv_heads * head_dim, hidden_size]
    // Reshape to [num_kv_heads, head_dim, hidden_size]
    let hidden_size = weight.dim(1)?;
    let w = weight.reshape((num_kv_heads, head_dim, hidden_size))?;
    // Expand: [num_kv_heads, 1, head_dim, hidden_size] -> repeat -> [num_kv_heads, group_size, head_dim, hidden_size]
    let w = w.unsqueeze(1)?;
    let w = w.expand((num_kv_heads, group_size, head_dim, hidden_size))?;
    // Reshape to [num_q_heads * head_dim, hidden_size]
    w.reshape((num_q_heads * head_dim, hidden_size))
}

/// Add symmetry-breaking noise to expanded KV weights.
/// After repeat_interleave, heads within a group are identical — adding small
/// Gaussian noise breaks this symmetry for DeltaNet state dynamics.
pub fn add_symmetry_noise(weight: &Tensor, sigma: f64) -> Result<Tensor> {
    let noise = Tensor::randn(0.0f32, sigma as f32, weight.shape(), weight.device())?
        .to_dtype(weight.dtype())?;
    weight + noise
}

// ---------------------------------------------------------------------------
// Qwen3.5 GGUF Weight Loader
// ---------------------------------------------------------------------------

/// Typed tensor access for Qwen3.5 GGUF models.
///
/// Wraps a `GgufFile` and provides methods to load each tensor by its
/// semantic name, mapping to the correct GGUF tensor path (e.g.
/// `blk.{layer}.ssm_a.weight` for the SSM A parameter).
///
/// All methods dequantize to f32 Candle tensors on the specified device.
pub struct Qwen35WeightLoader {
    gguf: GgufFile,
    config: Qwen35Config,
}

impl Qwen35WeightLoader {
    /// Open a GGUF file and build a Qwen35Config from its metadata.
    pub fn from_gguf(path: impl AsRef<Path>) -> std::result::Result<Self, String> {
        let gguf = GgufFile::open(path.as_ref())
            .map_err(|e| format!("Failed to open GGUF: {}", e))?;
        let config = Qwen35Config::from_gguf(&gguf)?;
        Ok(Self { gguf, config })
    }

    /// Access the parsed config.
    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }

    /// Access the underlying GgufFile.
    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    /// Consume the loader and return the underlying GgufFile.
    pub fn into_gguf(self) -> GgufFile {
        self.gguf
    }

    // --- Global tensors ---

    /// `token_embd.weight` — embedding table [hidden_size, vocab_size]
    pub fn embed_tokens(&self, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor("token_embd.weight", device)
    }

    /// `output_norm.weight` — final RMSNorm [hidden_size]
    pub fn output_norm(&self, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor("output_norm.weight", device)
    }

    /// `output.weight` — LM head projection [hidden_size, vocab_size]
    pub fn lm_head(&self, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor("output.weight", device)
    }

    // --- Per-layer GDN (SSM) tensors ---

    /// `blk.{layer}.ssm_a` (note: no `.weight` suffix in GGUF)
    pub fn ssm_a(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ssm_a", layer), device)
    }

    /// `blk.{layer}.ssm_conv1d.weight`
    pub fn ssm_conv1d(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ssm_conv1d.weight", layer), device)
    }

    /// `blk.{layer}.ssm_dt.bias`
    pub fn ssm_dt_bias(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ssm_dt.bias", layer), device)
    }

    /// `blk.{layer}.ssm_beta.weight`
    pub fn ssm_beta(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ssm_beta.weight", layer), device)
    }

    /// `blk.{layer}.ssm_alpha.weight`
    pub fn ssm_alpha(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ssm_alpha.weight", layer), device)
    }

    /// `blk.{layer}.ssm_norm.weight`
    pub fn ssm_norm(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ssm_norm.weight", layer), device)
    }

    /// `blk.{layer}.ssm_out.weight`
    pub fn ssm_out(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ssm_out.weight", layer), device)
    }

    // --- Per-layer GDN combined QKV + gate ---

    /// `blk.{layer}.attn_qkv.weight` — combined QKV for GDN layers
    pub fn attn_qkv(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.attn_qkv.weight", layer), device)
    }

    /// `blk.{layer}.attn_gate.weight` — gate (z) for GDN layers
    pub fn attn_gate(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.attn_gate.weight", layer), device)
    }

    // --- Per-layer attention (full) tensors ---

    /// `blk.{layer}.attn_q.weight`
    pub fn attn_q(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.attn_q.weight", layer), device)
    }

    /// `blk.{layer}.attn_k.weight`
    pub fn attn_k(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.attn_k.weight", layer), device)
    }

    /// `blk.{layer}.attn_v.weight`
    pub fn attn_v(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.attn_v.weight", layer), device)
    }

    /// `blk.{layer}.attn_output.weight`
    pub fn attn_out(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.attn_output.weight", layer), device)
    }

    /// `blk.{layer}.attn_q_norm.weight`
    pub fn attn_q_norm(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.attn_q_norm.weight", layer), device)
    }

    /// `blk.{layer}.attn_k_norm.weight`
    pub fn attn_k_norm(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.attn_k_norm.weight", layer), device)
    }

    // --- Per-layer common tensors (both GDN and attention) ---

    /// `blk.{layer}.attn_norm.weight` — pre-attention RMSNorm
    pub fn attn_norm(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.attn_norm.weight", layer), device)
    }

    /// `blk.{layer}.post_attention_norm.weight` — post-attention RMSNorm
    pub fn attn_post_norm(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.post_attention_norm.weight", layer), device)
    }

    /// `blk.{layer}.ffn_gate.weight` — FFN gate projection
    pub fn ffn_gate(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ffn_gate.weight", layer), device)
    }

    /// `blk.{layer}.ffn_up.weight` — FFN up projection
    pub fn ffn_up(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ffn_up.weight", layer), device)
    }

    /// `blk.{layer}.ffn_down.weight` — FFN down projection
    pub fn ffn_down(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ffn_down.weight", layer), device)
    }

    // --- Per-layer MoE FFN tensors (35B-A3B only) ---

    /// `blk.{layer}.ffn_gate_inp.weight` — MoE router gate: [hidden, num_experts]
    ///
    /// This is an F32 tensor (small, not quantized) that computes expert scores.
    pub fn moe_gate_inp(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ffn_gate_inp.weight", layer), device)
    }

    /// `blk.{layer}.ffn_gate_inp_shexp.weight` — shared expert gate bias: [hidden]
    pub fn moe_gate_inp_shexp(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ffn_gate_inp_shexp.weight", layer), device)
    }

    /// `blk.{layer}.ffn_gate_exps.weight` — routed expert gate weights: [hidden, expert_ffn, num_experts]
    ///
    /// Loaded as a regular Tensor (dequantized to F32) since the 3D shape is not
    /// directly supported by QMatMul.  Individual expert slices are extracted at
    /// forward time.
    pub fn moe_gate_exps(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ffn_gate_exps.weight", layer), device)
    }

    /// `blk.{layer}.ffn_up_exps.weight` — routed expert up-projection weights: [hidden, expert_ffn, num_experts]
    pub fn moe_up_exps(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ffn_up_exps.weight", layer), device)
    }

    /// `blk.{layer}.ffn_down_exps.weight` — routed expert down-projection weights: [expert_ffn, hidden, num_experts]
    pub fn moe_down_exps(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ffn_down_exps.weight", layer), device)
    }

    /// `blk.{layer}.ffn_gate_shexp.weight` — shared expert gate projection: [hidden, expert_ffn]
    pub fn moe_gate_shexp(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ffn_gate_shexp.weight", layer), device)
    }

    /// `blk.{layer}.ffn_up_shexp.weight` — shared expert up-projection: [hidden, expert_ffn]
    pub fn moe_up_shexp(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ffn_up_shexp.weight", layer), device)
    }

    /// `blk.{layer}.ffn_down_shexp.weight` — shared expert down-projection: [expert_ffn, hidden]
    pub fn moe_down_shexp(&self, layer: usize, device: &Device) -> Result<Tensor> {
        self.gguf.load_tensor(&format!("blk.{}.ffn_down_shexp.weight", layer), device)
    }

    /// Returns `true` if the given layer uses MoE FFN tensors (rather than a dense FFN).
    ///
    /// Detection is based on the presence of `blk.{layer}.ffn_gate_inp.weight` in the GGUF.
    pub fn layer_is_moe(&self, layer: usize) -> bool {
        self.has_tensor(&format!("blk.{}.ffn_gate_inp.weight", layer))
    }

    // --- MTP (multi-token prediction) tensors ---
    // These are in blk.{num_main_layers}.nextn.* (e.g. blk.64.nextn.*)

    /// `blk.{mtp_layer}.nextn.eh_proj.weight` — [2*hidden, hidden]
    pub fn mtp_eh_proj(&self, device: &Device) -> Result<Tensor> {
        let layer = self.config.num_main_layers;
        self.gguf.load_tensor(&format!("blk.{}.nextn.eh_proj.weight", layer), device)
    }

    /// `blk.{mtp_layer}.nextn.enorm.weight`
    pub fn mtp_enorm(&self, device: &Device) -> Result<Tensor> {
        let layer = self.config.num_main_layers;
        self.gguf.load_tensor(&format!("blk.{}.nextn.enorm.weight", layer), device)
    }

    /// `blk.{mtp_layer}.nextn.hnorm.weight`
    pub fn mtp_hnorm(&self, device: &Device) -> Result<Tensor> {
        let layer = self.config.num_main_layers;
        self.gguf.load_tensor(&format!("blk.{}.nextn.hnorm.weight", layer), device)
    }

    /// `blk.{mtp_layer}.nextn.shared_head_norm.weight`
    pub fn mtp_shared_head_norm(&self, device: &Device) -> Result<Tensor> {
        let layer = self.config.num_main_layers;
        self.gguf.load_tensor(&format!("blk.{}.nextn.shared_head_norm.weight", layer), device)
    }

    /// `blk.{mtp_layer}.nextn.attn_norm.weight` (if present)
    pub fn mtp_attn_norm(&self, device: &Device) -> Result<Tensor> {
        let layer = self.config.num_main_layers;
        self.gguf.load_tensor(&format!("blk.{}.nextn.attn_norm.weight", layer), device)
    }

    /// Check whether a tensor name exists in the GGUF file.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.gguf.get_tensor_info(name).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_expand_kv_weight() -> Result<()> {
        let device = Device::Cpu;
        // 4 KV heads, 20 Q heads, head_dim=4, hidden=8
        // weight: [4*4, 8] = [16, 8]
        let w = Tensor::randn(0.0f32, 1.0, (16, 8), &device)?;
        let expanded = expand_kv_weight(&w, 4, 20, 4)?;
        assert_eq!(expanded.dims(), &[80, 8], "Should expand to [20*4, 8]");

        // Verify group structure: heads 0-4 should be identical (from KV head 0)
        let h0: Vec<f32> = expanded.narrow(0, 0, 4)?.flatten_all()?.to_vec1()?;
        let h1: Vec<f32> = expanded.narrow(0, 4, 4)?.flatten_all()?.to_vec1()?;
        assert_eq!(h0, h1, "Heads in same group should be identical before noise");
        println!("KV expansion: [16,8] → {:?}", expanded.dims());
        Ok(())
    }

    #[test]
    fn test_expand_kv_weight_identity() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::randn(0.0f32, 1.0, (80, 8), &device)?;
        let expanded = expand_kv_weight(&w, 20, 20, 4)?;
        let orig: Vec<f32> = w.flatten_all()?.to_vec1()?;
        let exp: Vec<f32> = expanded.flatten_all()?.to_vec1()?;
        assert_eq!(orig, exp, "Same head count should be identity");
        Ok(())
    }

    #[test]
    fn test_symmetry_noise() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::ones((10, 10), DType::F32, &device)?;
        let noisy = add_symmetry_noise(&w, 0.01)?;
        // Should be close to 1.0 but not exactly
        let vals: Vec<f32> = noisy.flatten_all()?.to_vec1()?;
        let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
        assert!((mean - 1.0).abs() < 0.1, "Mean should be close to 1.0, got {}", mean);
        // Not all identical
        let all_same = vals.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
        assert!(!all_same, "Noise should break symmetry");
        println!("Symmetry noise: mean={:.4}, std~0.01", mean);
        Ok(())
    }

    #[test]
    fn test_load_nanbeige_weights() -> Result<()> {
        let model_dir = format!(
            "{}/.chimere/models/Nanbeige4.1-3B",
            std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into())
        );
        let path = std::path::Path::new(&model_dir);
        if !path.exists() {
            println!("Skipping: Nanbeige model not found");
            return Ok(());
        }

        let config = ChimereModelConfig::nanbeige_3b();
        let loader = WeightLoader::from_dir(path, config, DType::F32, &Device::Cpu)?;

        // Test loading various weights
        let embed = loader.embed_tokens_weight()?;
        assert_eq!(embed.dims(), &[166144, 2560], "embed_tokens shape");

        let lm_head = loader.lm_head_weight()?;
        assert_eq!(lm_head.dims(), &[166144, 2560], "lm_head shape");

        let norm = loader.final_norm_weight()?;
        assert_eq!(norm.dims(), &[2560], "final norm shape");

        // Layer 0 attention
        let attn_vb = loader.layer_attn_vb(0);
        let q_weight = attn_vb.get_with_hints(&[2560, 2560], "q_proj.weight", Default::default())?;
        assert_eq!(q_weight.dims(), &[2560, 2560], "q_proj shape");

        let k_weight = attn_vb.get_with_hints(&[512, 2560], "k_proj.weight", Default::default())?;
        assert_eq!(k_weight.dims(), &[512, 2560], "k_proj shape");

        // Layer 0 MLP
        let mlp_vb = loader.layer_mlp_vb(0);
        let gate = mlp_vb.get_with_hints(&[10496, 2560], "gate_proj.weight", Default::default())?;
        assert_eq!(gate.dims(), &[10496, 2560], "gate_proj shape");

        // Layernorms
        let ln = loader.input_layernorm_weight(0)?;
        assert_eq!(ln.dims(), &[2560], "input_layernorm shape");

        println!("Loaded Nanbeige weights: embed {:?}, q_proj {:?}, gate_proj {:?}",
            embed.dims(), q_weight.dims(), gate.dims());
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Qwen35WeightLoader tests
    // -----------------------------------------------------------------------

    fn test_gguf_path() -> String {
        let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());
        format!(
            "{}/.chimere/models/qwopus-27b/Qwen3.5-27B-Opus-IQ3S-MTP.gguf",
            home
        )
    }

    fn skip_if_missing(path: &str) -> bool {
        if !std::path::Path::new(path).exists() {
            eprintln!("Skipping test: GGUF file not found at {}", path);
            true
        } else {
            false
        }
    }

    #[test]
    fn test_qwen35_weight_loader() -> Result<()> {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return Ok(());
        }

        let loader = Qwen35WeightLoader::from_gguf(&path)
            .map_err(|e| candle_core::Error::Msg(e))?;

        let cfg = loader.config();
        assert_eq!(cfg.hidden_size, 5120, "hidden_size mismatch");
        assert_eq!(cfg.vocab_size, 248320, "vocab_size mismatch");

        // output_norm.weight is F32, always loadable
        let norm = loader.output_norm(&Device::Cpu)?;
        assert_eq!(norm.dims(), &[5120], "output_norm shape mismatch");

        // Verify GDN-layer tensors exist (layer 0 is GDN)
        assert!(loader.has_tensor("blk.0.attn_qkv.weight"), "blk.0.attn_qkv.weight should exist");
        assert!(loader.has_tensor("blk.0.attn_gate.weight"), "blk.0.attn_gate.weight should exist");
        assert!(loader.has_tensor("blk.0.ffn_gate.weight"), "blk.0.ffn_gate.weight should exist");
        assert!(loader.has_tensor("blk.0.attn_norm.weight"), "blk.0.attn_norm.weight should exist");

        // Verify attention-layer tensors exist (layer 3 is attention for Qwen3.5)
        assert!(loader.has_tensor("blk.3.attn_q.weight") || loader.has_tensor("blk.3.attn_qkv.weight"),
            "layer 3 should have attention tensors");

        // Verify MTP tensors exist
        assert!(loader.has_tensor("blk.64.nextn.eh_proj.weight"), "MTP eh_proj should exist");
        assert!(loader.has_tensor("blk.64.nextn.enorm.weight"), "MTP enorm should exist");
        assert!(loader.has_tensor("blk.64.nextn.hnorm.weight"), "MTP hnorm should exist");

        // Load attn_norm (should be F32 or BF16) and verify shape
        let attn_norm = loader.attn_norm(0, &Device::Cpu)?;
        assert_eq!(attn_norm.dims(), &[5120], "attn_norm shape mismatch");

        // Note: embed_tokens is IQ3_S (dequant not yet implemented),
        // so we verify its existence via has_tensor instead of loading.
        assert!(loader.has_tensor("token_embd.weight"), "token_embd.weight should exist");
        assert!(loader.has_tensor("output.weight"), "output.weight should exist");

        println!(
            "Qwen35WeightLoader: config=({}, {}, {}), norm={:?}, attn_norm={:?}",
            cfg.hidden_size, cfg.vocab_size, cfg.num_main_layers,
            norm.dims(),
            attn_norm.dims()
        );
        Ok(())
    }
}
