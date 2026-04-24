//! # GGUF File Parser & Tensor Loader — Chimère Engine
//!
//! Parses GGUF v3 files (the native format of llama.cpp) and provides
//! two loading paths:
//!
//! 1. **`load_tensor`**: CPU dequantization to F32 Candle Tensor (all types).
//! 2. **`load_qmatmul`**: GPU-native quantized loading via Candle's `QMatMul`.
//!    Types supported by Candle (Q8_0, Q4K, Q5K, Q6K, F32, F16, BF16, etc.)
//!    are uploaded directly to GPU in quantized form. Unsupported types (IQ3_S)
//!    fall back to CPU dequant → F32 on device.
//!
//! ## Supported quantization types
//!
//! | Type   | ID | CPU dequant | GPU QMatMul |
//! |--------|----|-------------|-------------|
//! | F32    |  0 | Full        | Native      |
//! | F16    |  1 | Full        | Native      |
//! | Q8_0   |  8 | Full        | Native      |
//! | Q4_K   | 12 | Full        | Native      |
//! | Q6_K   | 14 | Full        | Native      |
//! | IQ3_S  | 21 | Full        | F32 fallback|
//! | BF16   | 30 | Full        | Native      |

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{Device, Result as CandleResult, Tensor};
use half::f16;
use memmap2::Mmap;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

// ---------------------------------------------------------------------------
// GGUF value types (metadata KV)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// GGML quantization types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
    Tq3_1s = 44,
    Tq3_4s = 46,
}

impl GgmlType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2K),
            11 => Some(Self::Q3K),
            12 => Some(Self::Q4K),
            13 => Some(Self::Q5K),
            14 => Some(Self::Q6K),
            15 => Some(Self::Q8K),
            16 => Some(Self::Iq2Xxs),
            17 => Some(Self::Iq2Xs),
            18 => Some(Self::Iq3Xxs),
            19 => Some(Self::Iq1S),
            20 => Some(Self::Iq4Nl),
            21 => Some(Self::Iq3S),
            22 => Some(Self::Iq2S),
            23 => Some(Self::Iq4Xs),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::Iq1M),
            30 => Some(Self::Bf16),
            44 => Some(Self::Tq3_1s),
            46 => Some(Self::Tq3_4s),
            _ => None,
        }
    }

    /// Returns (block_size, type_size_in_bytes) for this quantization type.
    /// block_size = number of elements per quantization block.
    /// type_size  = number of bytes per block.
    pub fn block_info(&self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4_0 => (32, 18),   // 2 + 16
            Self::Q4_1 => (32, 20),   // 2 + 2 + 16
            Self::Q5_0 => (32, 22),   // 2 + 4 + 16
            Self::Q5_1 => (32, 24),   // 2 + 2 + 4 + 16
            Self::Q8_0 => (32, 34),   // 2 + 32
            Self::Q8_1 => (32, 40),   // 4 + 4 + 32
            Self::Q2K => (256, 84),   // 2 + 2 + 16 + 64
            Self::Q3K => (256, 110),  // 2 + 64 + 32 + 12
            Self::Q4K => (256, 144),  // 2 + 2 + 128 + 12
            Self::Q5K => (256, 176),  // 2 + 2 + 128 + 32 + 12
            Self::Q6K => (256, 210),  // 2 + 128 + 64 + 16
            Self::Q8K => (256, 292),  // 4 + 256 + 32
            Self::Iq2Xxs => (256, 66),
            Self::Iq2Xs => (256, 74),
            Self::Iq3Xxs => (256, 98),
            Self::Iq1S => (256, 50),
            Self::Iq4Nl => (32, 18),
            Self::Iq3S => (256, 110),  // 2 + 64 + 32 + 8 + 4
            Self::Iq2S => (256, 82),
            Self::Iq4Xs => (256, 136),
            Self::I8 => (1, 1),
            Self::I16 => (1, 2),
            Self::I32 => (1, 4),
            Self::I64 => (1, 8),
            Self::F64 => (1, 8),
            Self::Iq1M => (256, 56),
            Self::Bf16 => (1, 2),
            Self::Tq3_1s => (32, 16),  // 2*fp16 + 12 bytes 3-bit packed
            Self::Tq3_4s => (32, 16),  // 4 u8 E3M5 scales + 12 bytes 3-bit packed
        }
    }

    /// Number of bytes to store `n_elements` values of this type.
    pub fn tensor_bytes(&self, n_elements: usize) -> usize {
        let (block_size, type_size) = self.block_info();
        n_elements * type_size / block_size
    }
}

// ---------------------------------------------------------------------------
// Metadata value
// ---------------------------------------------------------------------------

/// A parsed GGUF metadata value.
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::Uint32(v) => Some(*v),
            // GGUF sometimes stores small non-negative integers as int32
            // (e.g. rope.dimension_sections = array[int32]).  Accept those too.
            GgufValue::Int32(v) if *v >= 0 => Some(*v as u32),
            GgufValue::Uint64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            GgufValue::Int64(v) if *v >= 0 && *v <= u32::MAX as i64 => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            GgufValue::Int32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::Uint64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::Float32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor info (parsed from header, before data section)
// ---------------------------------------------------------------------------

/// Information about a single tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    /// Dimensions in GGUF order (row-major, innermost first).
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Offset of this tensor's data *relative to the start of the data section*.
    pub data_offset: u64,
    /// Total number of elements.
    pub n_elements: usize,
}

// ---------------------------------------------------------------------------
// GGUF Reader
// ---------------------------------------------------------------------------

/// Parsed GGUF file with memory-mapped data for zero-copy tensor access.
pub struct GgufFile {
    _mmap: Mmap,
    /// Pointer to the start of the mmap'd data (kept for lifetime).
    data: *const u8,
    data_len: usize,

    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
    pub alignment: usize,

    /// Metadata key-value pairs, in file order.
    pub metadata: HashMap<String, GgufValue>,
    /// Tensor infos, in file order.
    pub tensors: Vec<GgufTensorInfo>,
    /// Name -> index into `tensors`.
    tensor_index: HashMap<String, usize>,

    /// Absolute byte offset where tensor data starts in the file.
    pub data_section_offset: usize,
}

// SAFETY: The Mmap is kept alive for the lifetime of GgufFile, and we only
// read from data through immutable references with correct alignment.
unsafe impl Send for GgufFile {}
unsafe impl Sync for GgufFile {}

impl GgufFile {
    /// Open and parse a GGUF file.  Only the header + metadata + tensor infos
    /// are parsed; tensor data is left memory-mapped and read on demand.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        let data = mmap.as_ptr();
        let data_len = mmap.len();

        let mut cursor = std::io::Cursor::new(&mmap[..]);

        // --- Header ---
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid GGUF magic: 0x{:08X} (expected 0x{:08X})", magic, GGUF_MAGIC),
            ));
        }

        let version = cursor.read_u32::<LittleEndian>()?;
        if version < 2 || version > 3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported GGUF version: {} (expected 2 or 3)", version),
            ));
        }

        let n_tensors = cursor.read_u64::<LittleEndian>()?;
        let n_kv = cursor.read_u64::<LittleEndian>()?;

        // --- Metadata KV pairs ---
        let mut metadata = HashMap::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = read_gguf_string(&mut cursor)?;
            let value = read_gguf_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        // Check for custom alignment
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        // --- Tensor infos ---
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        let mut tensor_index = HashMap::with_capacity(n_tensors as usize);

        for i in 0..n_tensors as usize {
            let name = read_gguf_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(cursor.read_u64::<LittleEndian>()?);
            }
            let raw_type = cursor.read_u32::<LittleEndian>()?;
            let ggml_type = GgmlType::from_u32(raw_type).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Unknown GGML type {} for tensor '{}'", raw_type, name),
                )
            })?;
            let data_offset = cursor.read_u64::<LittleEndian>()?;

            let n_elements: usize = dims.iter().map(|&d| d as usize).product();

            tensor_index.insert(name.clone(), i);
            tensors.push(GgufTensorInfo {
                name,
                dims,
                ggml_type,
                data_offset,
                n_elements,
            });
        }

        // --- Compute data section offset (aligned) ---
        let header_end = cursor.position() as usize;
        let padding = header_end % alignment;
        let data_section_offset = if padding == 0 {
            header_end
        } else {
            header_end + alignment - padding
        };

        Ok(GgufFile {
            _mmap: mmap,
            data,
            data_len,
            version,
            n_tensors,
            n_kv,
            alignment,
            metadata,
            tensors,
            tensor_index,
            data_section_offset,
        })
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Get a tensor info by name.
    pub fn get_tensor_info(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Get the raw bytes for a tensor's data (zero-copy from mmap).
    pub fn tensor_data(&self, info: &GgufTensorInfo) -> &[u8] {
        let start = self.data_section_offset + info.data_offset as usize;
        let n_bytes = info.ggml_type.tensor_bytes(info.n_elements);
        let end = start + n_bytes;
        assert!(
            end <= self.data_len,
            "Tensor '{}' data extends beyond file: {}+{} > {}",
            info.name, start, n_bytes, self.data_len
        );
        unsafe { std::slice::from_raw_parts(self.data.add(start), n_bytes) }
    }

    /// Dequantize a tensor to f32 and return as a Candle Tensor.
    ///
    /// The shape is taken from the GGUF tensor info dims (reversed to
    /// match Candle's row-major convention).
    pub fn load_tensor(
        &self,
        name: &str,
        device: &Device,
    ) -> CandleResult<Tensor> {
        let info = self.get_tensor_info(name).ok_or_else(|| {
            candle_core::Error::Msg(format!("Tensor '{}' not found in GGUF file", name))
        })?;

        let raw = self.tensor_data(info);

        // Try our custom dequant first; fall back to Candle's QTensor for supported types
        match dequantize(raw, info.ggml_type, info.n_elements) {
            Ok(f32_data) => {
                let shape: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
                Tensor::from_vec(f32_data, shape.as_slice(), device)
            }
            Err(_) if Self::to_candle_dtype(info.ggml_type).is_some() => {
                // Use Candle's built-in QTensor dequantization (supports Q5_K, Q3_K, etc.)
                let qmm = self.load_qmatmul(name, &Device::Cpu)?;
                let tensor = match qmm {
                    candle_core::quantized::QMatMul::Tensor(t) => t,
                    candle_core::quantized::QMatMul::TensorF16(t) =>
                        t.to_dtype(candle_core::DType::F32)?,
                    candle_core::quantized::QMatMul::QTensor(qt) =>
                        qt.dequantize(&Device::Cpu)?,
                };
                tensor.to_device(device)
            }
            Err(e) => Err(e),
        }
    }

    /// Map our GGUF GgmlType to Candle's GgmlDType.
    ///
    /// Returns None for types not supported by Candle's quantized module
    /// (e.g., IQ3_S, IQ2_S, etc.) — those must be CPU-dequantized to F32.
    pub fn to_candle_dtype(ggml_type: GgmlType) -> Option<GgmlDType> {
        match ggml_type {
            GgmlType::F32 => Some(GgmlDType::F32),
            GgmlType::F16 => Some(GgmlDType::F16),
            GgmlType::Bf16 => Some(GgmlDType::BF16),
            GgmlType::Q4_0 => Some(GgmlDType::Q4_0),
            GgmlType::Q4_1 => Some(GgmlDType::Q4_1),
            GgmlType::Q5_0 => Some(GgmlDType::Q5_0),
            GgmlType::Q5_1 => Some(GgmlDType::Q5_1),
            GgmlType::Q8_0 => Some(GgmlDType::Q8_0),
            GgmlType::Q8_1 => Some(GgmlDType::Q8_1),
            GgmlType::Q2K => Some(GgmlDType::Q2K),
            GgmlType::Q3K => Some(GgmlDType::Q3K),
            GgmlType::Q4K => Some(GgmlDType::Q4K),
            GgmlType::Q5K => Some(GgmlDType::Q5K),
            GgmlType::Q6K => Some(GgmlDType::Q6K),
            GgmlType::Q8K => Some(GgmlDType::Q8K),
            // IQ types, integer types, F64 — not supported by Candle's quantized module
            _ => None,
        }
    }

    /// Load a tensor as a Candle `QMatMul` for GPU-accelerated quantized matmul.
    ///
    /// For types supported by Candle's quantized module (Q8_0, Q4K, Q6K, F32, F16, BF16, etc.),
    /// the raw quantized bytes are uploaded directly to GPU as a `QTensor` — no CPU dequantization.
    ///
    /// For unsupported types (IQ3_S, etc.), falls back to CPU dequantization to F32,
    /// then wraps in `QMatMul::Tensor`.
    pub fn load_qmatmul(
        &self,
        name: &str,
        device: &Device,
    ) -> CandleResult<QMatMul> {
        let info = self.get_tensor_info(name).ok_or_else(|| {
            candle_core::Error::Msg(format!("Tensor '{}' not found in GGUF file", name))
        })?;

        let raw = self.tensor_data(info);
        // GGUF dims are innermost-first; Candle expects outermost-first.
        let shape: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();

        if let Some(candle_dtype) = Self::to_candle_dtype(info.ggml_type) {
            // Native quantized path: upload raw bytes to GPU as QTensor
            let data = Cow::Borrowed(raw);
            let storage = QStorage::from_data(data, device, candle_dtype)?;
            let qtensor = QTensor::new(storage, shape)?;
            QMatMul::from_qtensor(qtensor)
        } else {
            // Fallback: CPU dequant to F32, then wrap as regular Tensor on device
            let f32_data = dequantize(raw, info.ggml_type, info.n_elements)?;
            let tensor = Tensor::from_vec(f32_data, shape.as_slice(), device)?;
            Ok(QMatMul::Tensor(tensor))
        }
    }

    /// Load the raw encoded bytes of a tensor as a flat `U8` Candle Tensor on `device`.
    ///
    /// No dequantization is performed — the bytes are uploaded verbatim.  This is used
    /// for IQ3_S expert tensors: the raw blocks are placed on GPU and a custom CUDA kernel
    /// dequantises only the 8 active experts at inference time.
    ///
    /// Also returns the number of logical elements and the shape
    /// `(dims[0_rev], dims[1_rev], dims[2_rev])` for 3-D tensors (GGUF dims reversed to
    /// Candle row-major order).  Returns an error if the tensor is not found or not 3-D.
    pub fn load_tensor_u8(
        &self,
        name  : &str,
        device: &Device,
    ) -> CandleResult<(Tensor, usize, (usize, usize, usize))> {
        let info = self.get_tensor_info(name).ok_or_else(|| {
            candle_core::Error::Msg(format!("Tensor '{}' not found in GGUF file", name))
        })?;

        if info.dims.len() != 3 {
            return Err(candle_core::Error::Msg(format!(
                "load_tensor_u8: '{}' expected 3-D tensor, got {} dims",
                name, info.dims.len()
            )));
        }

        let raw = self.tensor_data(info);
        let n_bytes = raw.len();
        let n_elements = info.n_elements;

        // GGUF dims are innermost-first; reverse for Candle row-major.
        let dims_rev: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
        let shape3 = (dims_rev[0], dims_rev[1], dims_rev[2]);

        // Upload raw bytes to device as a flat U8 tensor — no copy on Cpu device.
        let raw_tensor = Tensor::from_vec(raw.to_vec(), (n_bytes,), device)?;

        Ok((raw_tensor, n_elements, shape3))
    }

    /// Load a tensor as raw U8 bytes — works for any dimensionality.
    /// Returns (raw_u8_tensor, n_elements, reversed_dims).
    pub fn load_tensor_u8_any(
        &self,
        name  : &str,
        device: &Device,
    ) -> CandleResult<(Tensor, usize, Vec<usize>)> {
        let info = self.get_tensor_info(name).ok_or_else(|| {
            candle_core::Error::Msg(format!("Tensor '{}' not found in GGUF file", name))
        })?;
        let raw = self.tensor_data(info);
        let n_bytes = raw.len();
        let n_elements = info.n_elements;
        let dims_rev: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
        let raw_tensor = Tensor::from_vec(raw.to_vec(), (n_bytes,), device)?;
        Ok((raw_tensor, n_elements, dims_rev))
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|t| t.name.as_str()).collect()
    }

    /// Get the GGML type of a tensor by name.
    pub fn tensor_ggml_type(&self, name: &str) -> Option<GgmlType> {
        self.get_tensor_info(name).map(|info| info.ggml_type)
    }
}

// ---------------------------------------------------------------------------
// GGUF string / value reading helpers
// ---------------------------------------------------------------------------

fn read_gguf_string(cursor: &mut std::io::Cursor<&[u8]>) -> std::io::Result<String> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let pos = cursor.position() as usize;
    let data = cursor.get_ref();
    if pos + len > data.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "String extends beyond file",
        ));
    }
    let s = std::str::from_utf8(&data[pos..pos + len])
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        .to_string();
    cursor.set_position((pos + len) as u64);
    Ok(s)
}

fn read_gguf_value(cursor: &mut std::io::Cursor<&[u8]>) -> std::io::Result<GgufValue> {
    let type_id = cursor.read_u32::<LittleEndian>()?;
    let vtype = GgufValueType::from_u32(type_id).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unknown GGUF value type: {}", type_id),
        )
    })?;
    read_gguf_typed_value(cursor, vtype)
}

fn read_gguf_typed_value(
    cursor: &mut std::io::Cursor<&[u8]>,
    vtype: GgufValueType,
) -> std::io::Result<GgufValue> {
    match vtype {
        GgufValueType::Uint8 => Ok(GgufValue::Uint8(cursor.read_u8()?)),
        GgufValueType::Int8 => Ok(GgufValue::Int8(cursor.read_i8()?)),
        GgufValueType::Uint16 => Ok(GgufValue::Uint16(cursor.read_u16::<LittleEndian>()?)),
        GgufValueType::Int16 => Ok(GgufValue::Int16(cursor.read_i16::<LittleEndian>()?)),
        GgufValueType::Uint32 => Ok(GgufValue::Uint32(cursor.read_u32::<LittleEndian>()?)),
        GgufValueType::Int32 => Ok(GgufValue::Int32(cursor.read_i32::<LittleEndian>()?)),
        GgufValueType::Float32 => Ok(GgufValue::Float32(cursor.read_f32::<LittleEndian>()?)),
        GgufValueType::Bool => {
            let b = cursor.read_u8()?;
            Ok(GgufValue::Bool(b != 0))
        }
        GgufValueType::String => {
            let s = read_gguf_string(cursor)?;
            Ok(GgufValue::String(s))
        }
        GgufValueType::Array => {
            let elem_type_id = cursor.read_u32::<LittleEndian>()?;
            let elem_type = GgufValueType::from_u32(elem_type_id).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Unknown array element type: {}", elem_type_id),
                )
            })?;
            let count = cursor.read_u64::<LittleEndian>()? as usize;
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(read_gguf_typed_value(cursor, elem_type)?);
            }
            Ok(GgufValue::Array(values))
        }
        GgufValueType::Uint64 => Ok(GgufValue::Uint64(cursor.read_u64::<LittleEndian>()?)),
        GgufValueType::Int64 => Ok(GgufValue::Int64(cursor.read_i64::<LittleEndian>()?)),
        GgufValueType::Float64 => Ok(GgufValue::Float64(cursor.read_f64::<LittleEndian>()?)),
    }
}

// ---------------------------------------------------------------------------
// Dequantization
// ---------------------------------------------------------------------------

/// Dequantize raw bytes to f32 values.
pub fn dequantize(data: &[u8], ggml_type: GgmlType, n_elements: usize) -> CandleResult<Vec<f32>> {
    match ggml_type {
        GgmlType::F32 => dequantize_f32(data, n_elements),
        GgmlType::F16 => dequantize_f16(data, n_elements),
        GgmlType::Bf16 => dequantize_bf16(data, n_elements),
        GgmlType::Q4_0 => dequantize_q4_0(data, n_elements),
        GgmlType::Q8_0 => dequantize_q8_0(data, n_elements),
        GgmlType::Q5K => dequantize_q5_k(data, n_elements),
        GgmlType::Q6K => dequantize_q6_k(data, n_elements),
        GgmlType::Q4K => dequantize_q4_k(data, n_elements),
        GgmlType::Iq3S => dequantize_iq3_s(data, n_elements),
        // For types with Candle QTensor support but no custom dequant,
        // fall through to the QMatMul-based path in the caller.
        // The caller should use load_qmatmul() + QMatMul::forward() for these types.
        other => Err(candle_core::Error::Msg(format!(
            "Dequantization not implemented for {:?}",
            other
        ))),
    }
}

/// F32: trivial reinterpret.
fn dequantize_f32(data: &[u8], n_elements: usize) -> CandleResult<Vec<f32>> {
    if data.len() < n_elements * 4 {
        return Err(candle_core::Error::Msg(format!(
            "F32 data too short: {} bytes for {} elements",
            data.len(),
            n_elements
        )));
    }
    let mut result = vec![0.0f32; n_elements];
    // SAFETY: f32 is 4 bytes, data is at least n_elements * 4 bytes.
    let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_elements) };
    result.copy_from_slice(src);
    Ok(result)
}

/// F16: convert half-precision floats to f32.
fn dequantize_f16(data: &[u8], n_elements: usize) -> CandleResult<Vec<f32>> {
    if data.len() < n_elements * 2 {
        return Err(candle_core::Error::Msg(format!(
            "F16 data too short: {} bytes for {} elements",
            data.len(),
            n_elements
        )));
    }
    let mut result = Vec::with_capacity(n_elements);
    let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, n_elements) };
    for &bits in src {
        result.push(f16::from_bits(bits).to_f32());
    }
    Ok(result)
}

/// BF16: convert bfloat16 to f32.
/// Q4_0: 18-byte blocks of 32 elements each.
/// Block: f16 scale (2 bytes) + 16 bytes of packed 4-bit quants (2 per byte).
/// Dequant: val = (nibble - 8) * scale
fn dequantize_q4_0(data: &[u8], n_elements: usize) -> CandleResult<Vec<f32>> {
    let block_size = 32usize;
    let bytes_per_block = 18usize;
    let n_blocks = n_elements / block_size;
    let mut out = vec![0.0f32; n_elements];
    for bi in 0..n_blocks {
        let block = &data[bi * bytes_per_block..(bi + 1) * bytes_per_block];
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        for j in 0..16 {
            let byte = block[2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out[bi * block_size + j] = lo as f32 * d;
            out[bi * block_size + j + 16] = hi as f32 * d;
        }
    }
    Ok(out)
}

/// BF16 layout: sign(1) + exponent(8) + mantissa(7).
/// To convert to f32, shift left by 16 bits.
fn dequantize_bf16(data: &[u8], n_elements: usize) -> CandleResult<Vec<f32>> {
    if data.len() < n_elements * 2 {
        return Err(candle_core::Error::Msg(format!(
            "BF16 data too short: {} bytes for {} elements",
            data.len(),
            n_elements
        )));
    }
    let mut result = Vec::with_capacity(n_elements);
    let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, n_elements) };
    for &bits in src {
        let f32_bits = (bits as u32) << 16;
        result.push(f32::from_bits(f32_bits));
    }
    Ok(result)
}

/// Q8_0: 34-byte blocks encoding 32 elements each.
///
/// Block layout:
///   - `d`: f16 scale (2 bytes)
///   - `qs[32]`: int8 quantized values (32 bytes)
///
/// Dequant: `result[i] = qs[i] * f16_to_f32(d)`
fn dequantize_q8_0(data: &[u8], n_elements: usize) -> CandleResult<Vec<f32>> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34; // 2 (f16 d) + 32 (i8 qs)

    if n_elements % BLOCK_SIZE != 0 {
        return Err(candle_core::Error::Msg(format!(
            "Q8_0: n_elements ({}) not divisible by block size ({})",
            n_elements, BLOCK_SIZE
        )));
    }

    let n_blocks = n_elements / BLOCK_SIZE;
    if data.len() < n_blocks * BLOCK_BYTES {
        return Err(candle_core::Error::Msg(format!(
            "Q8_0 data too short: {} bytes for {} blocks",
            data.len(),
            n_blocks
        )));
    }

    let mut result = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * BLOCK_BYTES..(block_idx + 1) * BLOCK_BYTES];

        // d: f16 stored as 2 little-endian bytes
        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let d = f16::from_bits(d_bits).to_f32();

        // qs[32]: signed 8-bit integers
        for i in 0..BLOCK_SIZE {
            let q = block[2 + i] as i8;
            result.push(q as f32 * d);
        }
    }

    Ok(result)
}

/// Q6_K: super-blocks of 256 elements.
///
/// Block layout (210 bytes total):
///   - `ql[128]`:   lower 4 bits of 6-bit quants (uint8)
///   - `qh[64]`:    upper 2 bits of 6-bit quants (uint8)
///   - `scales[16]`: per-sub-block scales (int8)
///   - `d`:          super-block scale (f16, 2 bytes)
///
/// The 256 elements are processed in 2 groups of 128.
/// Each group of 128 is split into 4 sub-groups of 32:
///   - sub-group 0: elements [0..32)    from ql[0..32] low nibble  + qh bits [0..1]
///   - sub-group 1: elements [32..64)   from ql[32..64] low nibble + qh bits [2..3]
///   - sub-group 2: elements [64..96)   from ql[0..32] high nibble + qh bits [4..5]
///   - sub-group 3: elements [96..128)  from ql[32..64] high nibble + qh bits [6..7]
///
/// Dequant: `result[i] = d * scale[sub_block] * (q6 - 32)`
fn dequantize_q6_k(data: &[u8], n_elements: usize) -> CandleResult<Vec<f32>> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 210; // 128 + 64 + 16 + 2

    if n_elements % QK_K != 0 {
        return Err(candle_core::Error::Msg(format!(
            "Q6_K: n_elements ({}) not divisible by {}",
            n_elements, QK_K
        )));
    }

    let n_blocks = n_elements / QK_K;
    if data.len() < n_blocks * BLOCK_BYTES {
        return Err(candle_core::Error::Msg(format!(
            "Q6_K data too short: {} bytes for {} blocks",
            data.len(),
            n_blocks
        )));
    }

    let mut result = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * BLOCK_BYTES..(block_idx + 1) * BLOCK_BYTES];

        // Parse block layout:
        //   ql[128] at offset 0
        //   qh[64]  at offset 128
        //   scales[16] at offset 192
        //   d (f16) at offset 208
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d_bits = u16::from_le_bytes([block[208], block[209]]);
        let d = f16::from_bits(d_bits).to_f32();

        // Process 2 groups of 128 elements
        let mut out = [0.0f32; QK_K];
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        let mut y_off = 0usize;

        for _group in 0..2 {
            // Within each group of 128, iterate over 32 positions
            for l in 0..32 {
                let is = l / 16; // scale index offset within group (0 or 1)

                let q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) as i8 - 32;
                let q2 =
                    ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 =
                    ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i8 - 32;

                let sc0 = scales[sc_off + is] as i8;
                let sc2 = scales[sc_off + is + 2] as i8;
                let sc4 = scales[sc_off + is + 4] as i8;
                let sc6 = scales[sc_off + is + 6] as i8;

                out[y_off + l] = d * sc0 as f32 * q1 as f32;
                out[y_off + l + 32] = d * sc2 as f32 * q2 as f32;
                out[y_off + l + 64] = d * sc4 as f32 * q3 as f32;
                out[y_off + l + 96] = d * sc6 as f32 * q4 as f32;
            }

            y_off += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }

        result.extend_from_slice(&out);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// IQ3_S lookup table (from llama.cpp ggml-common.h)
// ---------------------------------------------------------------------------

/// `iq3s_grid[512]`: each u32 encodes 4 unsigned byte values that represent
/// the base (unsigned) 3-bit quant levels for a group of 4 elements.
/// Index: 9-bit value formed from `qs[i]` (low 8 bits) | high-bit from `qh`.
#[rustfmt::skip]
const IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

// ---------------------------------------------------------------------------
// IQ3_S dequantization (type 21)
// ---------------------------------------------------------------------------

/// Dequantize IQ3_S: importance-weighted 3-bit quantization with lookup table.
///
/// Block layout (110 bytes per 256 elements):
///   - `d`:        f16 scale (2 bytes)
///   - `qs[64]`:   packed quant indices (64 bytes)
///   - `qh[8]`:    high bits for 9-bit grid index (8 bytes)
///   - `signs[32]`: sign bits (32 bytes)
///   - `scales[4]`: 2-bit sub-block scales packed as nibbles (4 bytes)
///
/// Ported faithfully from llama.cpp `dequantize_row_iq3_s`.
fn dequantize_iq3_s(data: &[u8], n_elements: usize) -> CandleResult<Vec<f32>> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 110; // 2 + 64 + 8 + 32 + 4

    if n_elements % QK_K != 0 {
        return Err(candle_core::Error::Msg(format!(
            "IQ3_S: n_elements ({}) not divisible by {}",
            n_elements, QK_K
        )));
    }

    let n_blocks = n_elements / QK_K;
    if data.len() < n_blocks * BLOCK_BYTES {
        return Err(candle_core::Error::Msg(format!(
            "IQ3_S data too short: {} bytes for {} blocks",
            data.len(),
            n_blocks
        )));
    }

    // kmask_iq2xs: bit masks for sign extraction [1, 2, 4, 8, 16, 32, 64, 128]
    const KMASK: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

    let mut result = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * BLOCK_BYTES..];

        // d: f16 scale at offset 0
        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let d = f16::from_bits(d_bits).to_f32();

        // Block offsets:
        //   qs:     offset 2,  64 bytes
        //   qh:     offset 66,  8 bytes
        //   signs:  offset 74, 32 bytes
        //   scales: offset 106, 4 bytes
        let qs_base = 2;
        let qh_base = 66;
        let signs_base = 74;
        let scales_base = 106;

        let mut qs_off = qs_base;
        let mut qh_off = qh_base;
        let mut signs_off = signs_base;

        // Process 8 sub-blocks of 32 elements each, but the C code processes
        // them in pairs (ib32 += 2), so 4 pairs.
        // Each pair = 2 sub-blocks of 32 = 64 elements.
        for ib32 in (0..8).step_by(2) {
            // 2-bit scales packed as nibbles:
            //   db1 uses low nibble of scales[ib32/2]
            //   db2 uses high nibble of scales[ib32/2]
            let scale_byte = block[scales_base + ib32 / 2];
            let db1 = d * (1 + 2 * (scale_byte & 0x0F) as u32) as f32;
            let db2 = d * (1 + 2 * (scale_byte >> 4) as u32) as f32;

            // --- First sub-block of 32 elements (uses db1) ---
            for l in 0..4 {
                // 9-bit grid index: qs[2*l+0] | high bit from qh[0]
                let grid_idx1 =
                    block[qs_off + 2 * l] as usize
                    | (((block[qh_off] as usize) << (8 - 2 * l)) & 256);
                let grid_idx2 =
                    block[qs_off + 2 * l + 1] as usize
                    | (((block[qh_off] as usize) << (7 - 2 * l)) & 256);

                // Look up 4 unsigned byte values from grid
                let grid1_u32 = IQ3S_GRID[grid_idx1];
                let grid2_u32 = IQ3S_GRID[grid_idx2];
                let grid1 = grid1_u32.to_le_bytes();
                let grid2 = grid2_u32.to_le_bytes();

                let sign_byte = block[signs_off + l];

                // Output 8 elements: 4 from grid1 + 4 from grid2, ALL using db1
                for j in 0..4 {
                    let sign = if sign_byte & KMASK[j] != 0 { -1.0f32 } else { 1.0f32 };
                    result.push(db1 * grid1[j] as f32 * sign);
                }
                for j in 0..4 {
                    let sign = if sign_byte & KMASK[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
                    result.push(db1 * grid2[j] as f32 * sign);
                }
            }
            qs_off += 8;
            signs_off += 4;

            // --- Second sub-block of 32 elements (uses db2) ---
            for l in 0..4 {
                let grid_idx1 =
                    block[qs_off + 2 * l] as usize
                    | (((block[qh_off + 1] as usize) << (8 - 2 * l)) & 256);
                let grid_idx2 =
                    block[qs_off + 2 * l + 1] as usize
                    | (((block[qh_off + 1] as usize) << (7 - 2 * l)) & 256);

                let grid1_u32 = IQ3S_GRID[grid_idx1];
                let grid2_u32 = IQ3S_GRID[grid_idx2];
                let grid1 = grid1_u32.to_le_bytes();
                let grid2 = grid2_u32.to_le_bytes();

                let sign_byte = block[signs_off + l];

                for j in 0..4 {
                    let sign = if sign_byte & KMASK[j] != 0 { -1.0f32 } else { 1.0f32 };
                    result.push(db2 * grid1[j] as f32 * sign);
                }
                for j in 0..4 {
                    let sign = if sign_byte & KMASK[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
                    result.push(db2 * grid2[j] as f32 * sign);
                }
            }
            qh_off += 2;
            qs_off += 8;
            signs_off += 4;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Q4_K dequantization (type 12)
// ---------------------------------------------------------------------------

/// Extract scale and min for sub-block `j` from the 12-byte `scales` array.
///
/// Faithfully ported from llama.cpp `get_scale_min_k4`.
/// For j < 4:  scale = scales[j] & 63,  min = scales[j+4] & 63
/// For j >= 4: scale = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
///             min   = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// Dequantize Q5_K: 5-bit quantization with per-sub-block scales and mins.
///
/// Block layout (176 bytes per 256 elements):
///   - `d`:          f16 super-block scale (2 bytes)
///   - `dmin`:       f16 super-block min (2 bytes)
///   - `scales[12]`: sub-block scales+mins packed with 6 bits (12 bytes)
///   - `qh[32]`:     high bits (bit 4) for each of 256 elements (32 bytes)
///   - `qs[128]`:    low 4 bits, packed as nibbles (128 bytes)
///
/// Ported faithfully from llama.cpp `dequantize_row_q5_K`.
fn dequantize_q5_k(data: &[u8], n_elements: usize) -> CandleResult<Vec<f32>> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 176; // 2 + 2 + 12 + 32 + 128

    if n_elements % QK_K != 0 {
        return Err(candle_core::Error::Msg(format!(
            "Q5_K: n_elements ({}) not divisible by {}",
            n_elements, QK_K
        )));
    }

    let n_blocks = n_elements / QK_K;
    if data.len() < n_blocks * BLOCK_BYTES {
        return Err(candle_core::Error::Msg(format!(
            "Q5_K data too short: {} bytes for {} blocks",
            data.len(),
            n_blocks
        )));
    }

    let mut result = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * BLOCK_BYTES..];

        // d: f16 super-block scale at offset 0
        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let d = f16::from_bits(d_bits).to_f32();

        // dmin: f16 super-block min at offset 2
        let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
        let dmin = f16::from_bits(dmin_bits).to_f32();

        // scales[12] at offset 4
        let scales = &block[4..16];

        // qh[32] at offset 16: high bits for all 256 elements
        let qh_base = 16;

        // qs[128] at offset 48: low 4 bits packed as nibbles
        let qs_base = 48;

        // Process 256 elements in 4 groups of 64.
        // Each group reads 32 bytes of ql (low nibbles + high nibbles)
        // and uses shifting bit masks u1/u2 to extract the 5th bit from qh[32].
        //
        // Faithfully ported from ggml: dequantize_row_q5_K.
        // qh[l] stores 8 high-bits (one per bit position). The mask u1/u2
        // selects which bit to read for each of the 4 groups:
        //   group 0: u1=1 (bit 0), u2=2 (bit 1)
        //   group 1: u1=4 (bit 2), u2=8 (bit 3)
        //   group 2: u1=16 (bit 4), u2=32 (bit 5)
        //   group 3: u1=64 (bit 6), u2=128 (bit 7)
        let mut is = 0usize;
        let mut ql_off = qs_base;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        for _j in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;

            // First 32: low nibble + high bit from qh
            for l in 0..32 {
                let ql = block[ql_off + l] & 0xF;
                let qh_high = if block[qh_base + l] & u1 != 0 { 16u8 } else { 0u8 };
                result.push(d1 * (ql + qh_high) as f32 - min1);
            }
            // Second 32: high nibble + high bit from qh
            for l in 0..32 {
                let ql = block[ql_off + l] >> 4;
                let qh_high = if block[qh_base + l] & u2 != 0 { 16u8 } else { 0u8 };
                result.push(d2 * (ql + qh_high) as f32 - min2);
            }

            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    Ok(result)
}

/// Dequantize Q4_K: 4-bit quantization with per-sub-block scales and mins.
///
/// Block layout (144 bytes per 256 elements):
///   - `d`:          f16 super-block scale (2 bytes)
///   - `dmin`:       f16 super-block min (2 bytes)
///   - `scales[12]`: sub-block scales+mins packed with 6 bits (12 bytes)
///   - `qs[128]`:    4-bit quantized values (128 bytes)
///
/// Ported faithfully from llama.cpp `dequantize_row_q4_K`.
fn dequantize_q4_k(data: &[u8], n_elements: usize) -> CandleResult<Vec<f32>> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 144; // 2 + 2 + 12 + 128

    if n_elements % QK_K != 0 {
        return Err(candle_core::Error::Msg(format!(
            "Q4_K: n_elements ({}) not divisible by {}",
            n_elements, QK_K
        )));
    }

    let n_blocks = n_elements / QK_K;
    if data.len() < n_blocks * BLOCK_BYTES {
        return Err(candle_core::Error::Msg(format!(
            "Q4_K data too short: {} bytes for {} blocks",
            data.len(),
            n_blocks
        )));
    }

    let mut result = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block = &data[block_idx * BLOCK_BYTES..];

        // d: f16 super-block scale at offset 0
        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let d = f16::from_bits(d_bits).to_f32();

        // dmin: f16 super-block min at offset 2
        let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
        let dmin = f16::from_bits(dmin_bits).to_f32();

        // scales[12] at offset 4
        let scales = &block[4..16];

        // qs[128] at offset 16
        let qs_base = 16;

        // Process 256 elements in groups of 64 (4 groups)
        let mut is = 0usize; // scale index
        let mut q_off = qs_base;

        for _j in 0..4 {
            // 4 groups of 64 elements
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;

            // First 32: low nibble
            for l in 0..32 {
                result.push(d1 * (block[q_off + l] & 0xF) as f32 - min1);
            }
            // Second 32: high nibble
            for l in 0..32 {
                result.push(d2 * (block[q_off + l] >> 4) as f32 - min2);
            }

            q_off += 32;
            is += 2;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    /// Path to the test GGUF file.
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
    fn test_parse_header() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }
        let gguf = GgufFile::open(&path).expect("Failed to open GGUF file");
        assert_eq!(gguf.version, 3, "GGUF version should be 3");
        assert!(
            gguf.n_tensors > 0,
            "Should have at least 1 tensor, got {}",
            gguf.n_tensors
        );
        // Qwen3.5-27B-Opus-IQ3S-MTP has 866 tensors
        assert_eq!(
            gguf.n_tensors, 866,
            "Expected 866 tensors, got {}",
            gguf.n_tensors
        );
        println!(
            "GGUF header: version={}, n_tensors={}, n_kv={}, alignment={}, data_offset={}",
            gguf.version, gguf.n_tensors, gguf.n_kv, gguf.alignment, gguf.data_section_offset
        );
    }

    #[test]
    fn test_read_metadata() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }
        let gguf = GgufFile::open(&path).expect("Failed to open GGUF file");

        // Architecture
        let arch = gguf
            .get_metadata("general.architecture")
            .expect("Missing general.architecture");
        let arch_str = arch.as_str().expect("architecture should be string");
        assert_eq!(arch_str, "qwen35", "Expected architecture 'qwen35', got '{}'", arch_str);

        // Block count: qwen35.block_count should be 65 (64 main + 1 nextn)
        // or it might just be the main block count
        let block_key = format!("{}.block_count", arch_str);
        let block_count = gguf
            .get_metadata(&block_key)
            .expect(&format!("Missing {}", block_key));
        let block_count_val = block_count.as_u32().expect("block_count should be u32");
        assert_eq!(
            block_count_val, 65,
            "Expected block_count=65, got {}",
            block_count_val
        );

        // Nextn predict layers
        let nextn_key = format!("{}.nextn_predict_layers", arch_str);
        let nextn = gguf
            .get_metadata(&nextn_key)
            .expect(&format!("Missing {}", nextn_key));
        let nextn_val = nextn.as_u32().expect("nextn_predict_layers should be u32");
        assert_eq!(nextn_val, 1, "Expected nextn_predict_layers=1, got {}", nextn_val);

        println!(
            "Metadata: arch={}, block_count={}, nextn_predict_layers={}",
            arch_str, block_count_val, nextn_val
        );

        // Print a few more metadata keys for debugging
        for key in [
            "general.name",
            &format!("{}.embedding_length", arch_str),
            &format!("{}.attention.head_count", arch_str),
            &format!("{}.attention.head_count_kv", arch_str),
        ] {
            if let Some(val) = gguf.get_metadata(key) {
                println!("  {}: {:?}", key, val);
            }
        }
    }

    #[test]
    fn test_list_tensors() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }
        let gguf = GgufFile::open(&path).expect("Failed to open GGUF file");

        let names = gguf.tensor_names();

        // Verify key tensors exist
        assert!(
            names.contains(&"token_embd.weight"),
            "Missing token_embd.weight"
        );
        assert!(
            names.contains(&"output_norm.weight"),
            "Missing output_norm.weight"
        );

        // Block 0 tensors
        let blk0_prefix = "blk.0.";
        let blk0_tensors: Vec<&&str> = names.iter().filter(|n| n.starts_with(blk0_prefix)).collect();
        assert!(
            !blk0_tensors.is_empty(),
            "No tensors found with prefix blk.0."
        );

        // Nextn tensors (blk.64.nextn.*)
        let nextn_prefix = "blk.64.nextn.";
        let nextn_tensors: Vec<&&str> =
            names.iter().filter(|n| n.starts_with(nextn_prefix)).collect();
        assert!(
            !nextn_tensors.is_empty(),
            "No nextn tensors found with prefix blk.64.nextn."
        );

        println!("Sample blk.0 tensors: {:?}", &blk0_tensors[..blk0_tensors.len().min(5)]);
        println!("Nextn tensors: {:?}", nextn_tensors);
        println!("Total tensor count: {}", names.len());
    }

    #[test]
    fn test_qwen35_layer_tensor_names() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }
        let gguf = GgufFile::open(&path).expect("Failed to open GGUF file");
        let names = gguf.tensor_names();

        // Layer 0 (GDN) — all tensors
        println!("=== blk.0 (GDN) ===");
        for n in &names {
            if n.starts_with("blk.0.") {
                let info = gguf.get_tensor_info(n).unwrap();
                println!("  {} dims={:?} type={:?}", n, info.dims, info.ggml_type);
            }
        }

        // Layer 3 (attention) — all tensors
        println!("=== blk.3 (attention) ===");
        for n in &names {
            if n.starts_with("blk.3.") {
                let info = gguf.get_tensor_info(n).unwrap();
                println!("  {} dims={:?} type={:?}", n, info.dims, info.ggml_type);
            }
        }

        // Global (non-blk) tensors
        println!("=== Global ===");
        for n in &names {
            if !n.starts_with("blk.") {
                let info = gguf.get_tensor_info(n).unwrap();
                println!("  {} dims={:?} type={:?}", n, info.dims, info.ggml_type);
            }
        }

        // MTP layer 64
        println!("=== blk.64 (MTP) ===");
        for n in &names {
            if n.starts_with("blk.64.") {
                let info = gguf.get_tensor_info(n).unwrap();
                println!("  {} dims={:?} type={:?}", n, info.dims, info.ggml_type);
            }
        }
    }

    #[test]
    fn test_tensor_shapes() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }
        let gguf = GgufFile::open(&path).expect("Failed to open GGUF file");

        // token_embd.weight: expected [embedding_length, vocab_size]
        // For Qwen3.5-27B: embedding_length=5120, vocab_size varies
        let embd = gguf
            .get_tensor_info("token_embd.weight")
            .expect("Missing token_embd.weight");
        println!(
            "token_embd.weight: dims={:?}, type={:?}, n_elements={}",
            embd.dims, embd.ggml_type, embd.n_elements
        );
        // GGUF dims are innermost-first, so for a [vocab, hidden] weight:
        // dims[0] = hidden = 5120 (or similar)
        assert_eq!(embd.dims.len(), 2, "token_embd should be 2D");

        // blk.0.ffn_gate.weight
        let ffn_gate = gguf.get_tensor_info("blk.0.ffn_gate.weight");
        if let Some(fg) = ffn_gate {
            println!(
                "blk.0.ffn_gate.weight: dims={:?}, type={:?}",
                fg.dims, fg.ggml_type
            );
        }

        // output_norm.weight should be 1D
        let output_norm = gguf
            .get_tensor_info("output_norm.weight")
            .expect("Missing output_norm.weight");
        println!(
            "output_norm.weight: dims={:?}, type={:?}",
            output_norm.dims, output_norm.ggml_type
        );
        assert_eq!(output_norm.dims.len(), 1, "output_norm should be 1D");
    }

    #[test]
    fn test_dequant_q8_0() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }
        let gguf = GgufFile::open(&path).expect("Failed to open GGUF file");

        // Find a Q8_0 tensor. Try nextn tensors first, they tend to be Q8_0.
        let q8_tensor = gguf
            .tensors
            .iter()
            .find(|t| t.ggml_type == GgmlType::Q8_0);

        let info = match q8_tensor {
            Some(t) => t,
            None => {
                eprintln!("No Q8_0 tensors found in this GGUF, skipping test");
                return;
            }
        };
        println!(
            "Testing Q8_0 dequant on '{}': dims={:?}, n_elements={}",
            info.name, info.dims, info.n_elements
        );

        let raw = gguf.tensor_data(info);
        let result = dequantize(raw, GgmlType::Q8_0, info.n_elements)
            .expect("Q8_0 dequantization failed");

        assert_eq!(result.len(), info.n_elements);

        // Values should be non-zero floats (not all zeros)
        let non_zero = result.iter().filter(|&&v| v.abs() > 1e-10).count();
        assert!(
            non_zero > info.n_elements / 10,
            "Too few non-zero values: {}/{}",
            non_zero,
            info.n_elements
        );

        // Values should be finite
        assert!(
            result.iter().all(|v| v.is_finite()),
            "Found non-finite values in Q8_0 dequantized data"
        );

        // Print some stats
        let min = result.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = result.iter().sum::<f32>() / result.len() as f32;
        println!(
            "  Q8_0 stats: min={:.6}, max={:.6}, mean={:.6}, non_zero={}/{}",
            min, max, mean, non_zero, info.n_elements
        );
    }

    #[test]
    fn test_dequant_f32() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }
        let gguf = GgufFile::open(&path).expect("Failed to open GGUF file");

        // output_norm.weight is typically F32 in quantized models
        let info = gguf
            .get_tensor_info("output_norm.weight")
            .expect("Missing output_norm.weight");

        println!(
            "Testing F32 dequant on 'output_norm.weight': dims={:?}, type={:?}, n_elements={}",
            info.dims, info.ggml_type, info.n_elements
        );

        // This tensor might be F32 or BF16 depending on quantization
        if info.ggml_type != GgmlType::F32 && info.ggml_type != GgmlType::Bf16 {
            eprintln!(
                "output_norm.weight is {:?}, not F32/BF16 -- testing with load_tensor instead",
                info.ggml_type
            );
        }

        let tensor = gguf
            .load_tensor("output_norm.weight", &Device::Cpu)
            .expect("Failed to load output_norm.weight");

        // Should be 1D with shape [embedding_length]
        assert_eq!(tensor.rank(), 1, "output_norm should be 1D");
        let vals: Vec<f32> = tensor.to_vec1().expect("Failed to convert to vec");
        assert!(!vals.is_empty(), "output_norm should not be empty");

        // Values should be finite and mostly non-zero
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "Found non-finite values"
        );
        let non_zero = vals.iter().filter(|&&v| v.abs() > 1e-10).count();
        assert!(
            non_zero > vals.len() / 2,
            "Too few non-zero values: {}/{}",
            non_zero,
            vals.len()
        );

        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "  output_norm stats: shape={:?}, min={:.6}, max={:.6}, non_zero={}/{}",
            tensor.dims(),
            min,
            max,
            non_zero,
            vals.len()
        );
    }

    #[test]
    fn test_dequant_q6_k() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }
        let gguf = GgufFile::open(&path).expect("Failed to open GGUF file");

        // Find a Q6_K tensor
        let q6k_tensor = gguf
            .tensors
            .iter()
            .find(|t| t.ggml_type == GgmlType::Q6K);

        let info = match q6k_tensor {
            Some(t) => t,
            None => {
                eprintln!("No Q6_K tensors found in this GGUF, skipping test");
                return;
            }
        };
        println!(
            "Testing Q6_K dequant on '{}': dims={:?}, n_elements={}",
            info.name, info.dims, info.n_elements
        );

        let raw = gguf.tensor_data(info);
        let result = dequantize(raw, GgmlType::Q6K, info.n_elements)
            .expect("Q6_K dequantization failed");

        assert_eq!(result.len(), info.n_elements);

        // Values should be finite
        assert!(
            result.iter().all(|v| v.is_finite()),
            "Found non-finite values in Q6_K dequantized data"
        );

        let non_zero = result.iter().filter(|&&v| v.abs() > 1e-10).count();
        let min = result.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = result.iter().sum::<f32>() / result.len() as f32;
        println!(
            "  Q6_K stats: min={:.6}, max={:.6}, mean={:.6}, non_zero={}/{}",
            min, max, mean, non_zero, info.n_elements
        );
    }

    #[test]
    fn test_load_tensor_as_candle() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }
        let gguf = GgufFile::open(&path).expect("Failed to open GGUF file");

        // Load output_norm as a Candle tensor
        let t = gguf
            .load_tensor("output_norm.weight", &Device::Cpu)
            .expect("Failed to load tensor");

        assert_eq!(t.dtype(), DType::F32, "Loaded tensor should be F32");
        assert_eq!(t.rank(), 1, "output_norm should be 1D");
        println!("Candle tensor: shape={:?}, dtype={:?}", t.dims(), t.dtype());
    }

    /// Verify that the unit dequantization of Q8_0 is mathematically correct
    /// by constructing a known block and checking the output.
    #[test]
    fn test_q8_0_known_block() {
        // Construct a synthetic Q8_0 block: d=1.0 (as f16), qs = [0, 1, 2, ..., 31]
        let d_f16 = f16::from_f32(1.0);
        let d_bytes = d_f16.to_bits().to_le_bytes();

        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&d_bytes);
        for i in 0..32u8 {
            block.push(i); // these will be interpreted as i8
        }

        let result = dequantize_q8_0(&block, 32).expect("Dequant failed");
        assert_eq!(result.len(), 32);

        // With d=1.0, result[i] should equal i (as i8, cast to f32)
        for i in 0..32 {
            let expected = i as i8 as f32 * 1.0;
            assert!(
                (result[i] - expected).abs() < 1e-3,
                "result[{}] = {}, expected {}",
                i,
                result[i],
                expected
            );
        }
    }

    /// Verify BF16 dequant with known values.
    #[test]
    fn test_bf16_known_values() {
        // BF16 for 1.0: sign=0, exp=01111111 (127), mantissa=0000000
        // = 0 01111111 0000000 = 0x3F80
        // BF16 for -2.0: sign=1, exp=10000000 (128), mantissa=0000000
        // = 1 10000000 0000000 = 0xC000
        let data: Vec<u8> = vec![
            0x80, 0x3F, // 1.0 in BF16 (LE)
            0x00, 0xC0, // -2.0 in BF16 (LE)
            0x00, 0x00, // 0.0 in BF16 (LE)
        ];

        let result = dequantize_bf16(&data, 3).expect("BF16 dequant failed");
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-6, "Expected 1.0, got {}", result[0]);
        assert!((result[1] - (-2.0)).abs() < 1e-6, "Expected -2.0, got {}", result[1]);
        assert!((result[2] - 0.0).abs() < 1e-6, "Expected 0.0, got {}", result[2]);
    }

    #[test]
    fn test_dequant_iq3_s() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }
        let gguf = GgufFile::open(&path).expect("Failed to open GGUF file");

        // blk.0.ffn_gate.weight is IQ3_S, dims=[5120, 17408]
        let info = gguf
            .get_tensor_info("blk.0.ffn_gate.weight")
            .expect("Missing blk.0.ffn_gate.weight");
        assert_eq!(
            info.ggml_type,
            GgmlType::Iq3S,
            "Expected IQ3_S type, got {:?}",
            info.ggml_type
        );
        assert_eq!(
            info.dims,
            vec![5120, 17408],
            "Unexpected dims: {:?}",
            info.dims
        );
        assert_eq!(info.n_elements, 5120 * 17408);

        let raw = gguf.tensor_data(info);
        let result = dequantize(raw, GgmlType::Iq3S, info.n_elements)
            .expect("IQ3_S dequantization failed");

        assert_eq!(result.len(), info.n_elements);

        // All values should be finite
        assert!(
            result.iter().all(|v| v.is_finite()),
            "Found non-finite values in IQ3_S dequantized data"
        );

        // >90% non-zero (IQ3_S grid values are odd numbers, so most are non-zero)
        let non_zero = result.iter().filter(|&&v| v.abs() > 1e-10).count();
        let non_zero_pct = non_zero as f64 / result.len() as f64 * 100.0;
        assert!(
            non_zero_pct > 90.0,
            "Too few non-zero values: {:.1}% ({}/{})",
            non_zero_pct,
            non_zero,
            result.len()
        );

        // Values should be in a reasonable range for weight tensors
        let min = result.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = result.iter().sum::<f32>() / result.len() as f32;
        let absmax = min.abs().max(max.abs());
        assert!(
            absmax < 10.0,
            "IQ3_S values out of expected range: min={}, max={}",
            min,
            max
        );

        println!(
            "  IQ3_S stats: min={:.6}, max={:.6}, mean={:.6}, non_zero={:.1}% ({}/{})",
            min, max, mean, non_zero_pct, non_zero, result.len()
        );
    }

    #[test]
    fn test_dequant_q4_k() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }
        let gguf = GgufFile::open(&path).expect("Failed to open GGUF file");

        // blk.3.attn_v.weight is Q4_K, dims=[5120, 1024]
        let info = gguf
            .get_tensor_info("blk.3.attn_v.weight")
            .expect("Missing blk.3.attn_v.weight");
        assert_eq!(
            info.ggml_type,
            GgmlType::Q4K,
            "Expected Q4_K type, got {:?}",
            info.ggml_type
        );
        assert_eq!(
            info.dims,
            vec![5120, 1024],
            "Unexpected dims: {:?}",
            info.dims
        );
        assert_eq!(info.n_elements, 5120 * 1024);

        let raw = gguf.tensor_data(info);
        let result = dequantize(raw, GgmlType::Q4K, info.n_elements)
            .expect("Q4_K dequantization failed");

        assert_eq!(result.len(), info.n_elements);

        // All values should be finite
        assert!(
            result.iter().all(|v| v.is_finite()),
            "Found non-finite values in Q4_K dequantized data"
        );

        // Should have substantial non-zero values
        let non_zero = result.iter().filter(|&&v| v.abs() > 1e-10).count();
        assert!(
            non_zero > result.len() / 2,
            "Too few non-zero values: {}/{}",
            non_zero,
            result.len()
        );

        let min = result.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = result.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = result.iter().sum::<f32>() / result.len() as f32;
        println!(
            "  Q4_K stats: min={:.6}, max={:.6}, mean={:.6}, non_zero={}/{}",
            min, max, mean, non_zero, result.len()
        );
    }

    #[test]
    fn test_load_full_layer() {
        let path = test_gguf_path();
        if skip_if_missing(&path) {
            return;
        }
        let gguf = GgufFile::open(&path).expect("Failed to open GGUF file");
        let dev = Device::Cpu;

        // Layer 0 is a GDN (recurrent) layer in Qwen3.5-27B.
        // Expected tensors and their shapes (GGUF dims are innermost-first,
        // Candle reverses to outermost-first):
        let expected: Vec<(&str, Vec<usize>)> = vec![
            ("blk.0.attn_gate.weight", vec![6144, 5120]),
            ("blk.0.attn_norm.weight", vec![5120]),
            ("blk.0.attn_qkv.weight", vec![10240, 5120]),
            ("blk.0.ffn_down.weight", vec![5120, 17408]),
            ("blk.0.ffn_gate.weight", vec![17408, 5120]),
            ("blk.0.ffn_up.weight", vec![17408, 5120]),
            ("blk.0.post_attention_norm.weight", vec![5120]),
            ("blk.0.ssm_a", vec![48]),
            ("blk.0.ssm_alpha.weight", vec![48, 5120]),
            ("blk.0.ssm_beta.weight", vec![48, 5120]),
            ("blk.0.ssm_conv1d.weight", vec![10240, 4]),
            ("blk.0.ssm_dt.bias", vec![48]),
            ("blk.0.ssm_norm.weight", vec![128]),
            ("blk.0.ssm_out.weight", vec![5120, 6144]),
        ];

        for (name, expected_shape) in &expected {
            let tensor = gguf
                .load_tensor(name, &dev)
                .unwrap_or_else(|e| panic!("Failed to load '{}': {}", name, e));

            let actual_shape: Vec<usize> = tensor.dims().to_vec();
            assert_eq!(
                &actual_shape, expected_shape,
                "Shape mismatch for '{}': got {:?}, expected {:?}",
                name, actual_shape, expected_shape
            );

            // Verify all values are finite
            let flat: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                flat.iter().all(|v| v.is_finite()),
                "Non-finite values in '{}'",
                name
            );

            println!(
                "  {} shape={:?} dtype={:?} OK",
                name,
                actual_shape,
                tensor.dtype()
            );
        }

        println!("All {} layer-0 tensors loaded successfully.", expected.len());
    }
}

