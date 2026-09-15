# Vendored from llama.cpp gguf-py (MIT). See LICENSE.gguf-py.
#
# The GGUF container types and block geometry are copied verbatim so the converter can decode
# reference checkpoint blocks without depending on an external gguf-py installation.

from __future__ import annotations

from enum import IntEnum

class GGMLQuantizationType(IntEnum):
    F32     = 0
    F16     = 1
    Q4_0    = 2
    Q4_1    = 3
    Q5_0    = 6
    Q5_1    = 7
    Q8_0    = 8
    Q8_1    = 9
    Q2_K    = 10
    Q3_K    = 11
    Q4_K    = 12
    Q5_K    = 13
    Q6_K    = 14
    Q8_K    = 15
    IQ2_XXS = 16
    IQ2_XS  = 17
    IQ3_XXS = 18
    IQ1_S   = 19
    IQ4_NL  = 20
    IQ3_S   = 21
    IQ2_S   = 22
    IQ4_XS  = 23
    I8      = 24
    I16     = 25
    I32     = 26
    I64     = 27
    F64     = 28
    IQ1_M   = 29
    BF16    = 30
    TQ1_0   = 34
    TQ2_0   = 35
    MXFP4   = 39
    NVFP4   = 40
    Q1_0    = 41
    Q2_0    = 42


class ExpertGatingFuncType(IntEnum):
    SOFTMAX       = 1
    SIGMOID       = 2
    SQRTSOFTPLUS  = 4


# TODO: add GGMLFileType from ggml_ftype in ggml.h


# from llama_ftype in llama.h
# ALL VALUES SHOULD BE THE SAME HERE AS THEY ARE OVER THERE.
class LlamaFileType(IntEnum):
    ALL_F32              = 0
    MOSTLY_F16           = 1   # except 1d tensors
    MOSTLY_Q4_0          = 2   # except 1d tensors
    MOSTLY_Q4_1          = 3   # except 1d tensors
    # MOSTLY_Q4_1_SOME_F16 = 4   # tok_embeddings.weight and output.weight are F16
    # MOSTLY_Q4_2        = 5   # support has been removed
    # MOSTLY_Q4_3        = 6   # support has been removed
    MOSTLY_Q8_0          = 7   # except 1d tensors
    MOSTLY_Q5_0          = 8   # except 1d tensors
    MOSTLY_Q5_1          = 9   # except 1d tensors
    MOSTLY_Q2_K          = 10  # except 1d tensors
    MOSTLY_Q3_K_S        = 11  # except 1d tensors
    MOSTLY_Q3_K_M        = 12  # except 1d tensors
    MOSTLY_Q3_K_L        = 13  # except 1d tensors
    MOSTLY_Q4_K_S        = 14  # except 1d tensors
    MOSTLY_Q4_K_M        = 15  # except 1d tensors
    MOSTLY_Q5_K_S        = 16  # except 1d tensors
    MOSTLY_Q5_K_M        = 17  # except 1d tensors
    MOSTLY_Q6_K          = 18  # except 1d tensors
    MOSTLY_IQ2_XXS       = 19  # except 1d tensors
    MOSTLY_IQ2_XS        = 20  # except 1d tensors
    MOSTLY_Q2_K_S        = 21  # except 1d tensors
    MOSTLY_IQ3_XS        = 22  # except 1d tensors
    MOSTLY_IQ3_XXS       = 23  # except 1d tensors
    MOSTLY_IQ1_S         = 24  # except 1d tensors
    MOSTLY_IQ4_NL        = 25  # except 1d tensors
    MOSTLY_IQ3_S         = 26  # except 1d tensors
    MOSTLY_IQ3_M         = 27  # except 1d tensors
    MOSTLY_IQ2_S         = 28  # except 1d tensors
    MOSTLY_IQ2_M         = 29  # except 1d tensors
    MOSTLY_IQ4_XS        = 30  # except 1d tensors
    MOSTLY_IQ1_M         = 31  # except 1d tensors
    MOSTLY_BF16          = 32  # except 1d tensors
    # MOSTLY_Q4_0_4_4      = 33  # removed from gguf files, use Q4_0 and runtime repack
    # MOSTLY_Q4_0_4_8      = 34  # removed from gguf files, use Q4_0 and runtime repack
    # MOSTLY_Q4_0_8_8      = 35  # removed from gguf files, use Q4_0 and runtime repack
    MOSTLY_TQ1_0         = 36  # except 1d tensors
    MOSTLY_TQ2_0         = 37  # except 1d tensors
    MOSTLY_MXFP4_MOE     = 38  # except 1d tensors
    MOSTLY_NVFP4         = 39  # except 1d tensors
    MOSTLY_Q1_0          = 40  # except 1d tensors
    MOSTLY_Q2_0          = 41  # except 1d tensors

    GUESSED              = 1024  # not specified in the model file


class GGUFEndian(IntEnum):
    LITTLE = 0
    BIG = 1


class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

    @staticmethod
    def get_type(val: Any) -> GGUFValueType:
        if isinstance(val, (str, bytes, bytearray)):
            return GGUFValueType.STRING
        elif isinstance(val, list):
            return GGUFValueType.ARRAY
        elif isinstance(val, float):
            return GGUFValueType.FLOAT32
        elif isinstance(val, bool):
            return GGUFValueType.BOOL
        elif isinstance(val, int):
            return GGUFValueType.INT32
        # TODO: need help with 64-bit types in Python
        else:
            raise ValueError(f"Unknown type: {type(val)}")


class VisionProjectorType:
    GEMMA3 = "gemma3"
    GEMMA3NV = "gemma3nv"
    GEMMA3NA = "gemma3na"
    GEMMA4V = "gemma4v"
    GEMMA4A = "gemma4a"
    GEMMA4UV = "gemma4uv" # "unified" variant
    GEMMA4UA = "gemma4ua" # "unified" variant
    PHI4 = "phi4"
    IDEFICS3 = "idefics3"
    PIXTRAL = "pixtral"
    LLAMA4 = "llama4"
    QWEN2VL = "qwen2vl_merger"
    QWEN25VL = "qwen2.5vl_merger"
    EXAONE4_5 = "exaone4_5"
    QWEN3VL = "qwen3vl_merger"
    STEP3VL = "step3vl"
    ULTRAVOX = "ultravox"
    INTERNVL = "internvl"
    QWEN2A = "qwen2a" # audio
    QWEN3A = "qwen3a" # audio
    GLMA = "glma" # audio
    QWEN25O = "qwen2.5o" # omni
    VOXTRAL = "voxtral"
    MERALION = "meralion"  # audio: Whisper + gated MLP adaptor
    LFM2 = "lfm2"
    KIMIVL = "kimivl"
    PADDLEOCR = "paddleocr"
    KIMIK25 = "kimik25"
    LIGHTONOCR = "lightonocr"
    COGVLM = "cogvlm"
    JANUS_PRO = "janus_pro"
    DOTSOCR = "dots_ocr"
    DEEPSEEKOCR = "deepseekocr"
    DEEPSEEKOCR2 = "deepseekocr2"
    LFM2A = "lfm2a" # audio
    MUSIC_FLAMINGO = "musicflamingo" # audio
    GLM4V = "glm4v"
    YOUTUVL = "youtuvl"
    NEMOTRON_V2_VL = "nemotron_v2_vl"
    QWEN3TTS_SPKENC = "qwen3tts_spkenc" # audio: ECAPA-TDNN speaker encoder
    QWEN3TTS_GEN = "qwen3tts_gen" # audio generation: code_predictor
    HUNYUANVL      = "hunyuanvl"
    PARAKEET       = "parakeet"  # audio
    MINIMAXM3      = "minimax_m3"
    MINICPMV4_6    = "minicpmv4_6"
    GRANITE_SPEECH = "granite_speech"  # audio
    MIMOVL         = "mimovl"
    MIMO_AUDIO     = "mimo_audio"
    GRANITE4_VISION = "granite4_vision"
    MUSE_GLIMMER   = "muse-glimmer"


# Items here are (block size, type size)
QK_K = 256
GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    GGMLQuantizationType.F32:     (1, 4),
    GGMLQuantizationType.F16:     (1, 2),
    GGMLQuantizationType.Q4_0:    (32, 2 + 16),
    GGMLQuantizationType.Q4_1:    (32, 2 + 2 + 16),
    GGMLQuantizationType.Q5_0:    (32, 2 + 4 + 16),
    GGMLQuantizationType.Q5_1:    (32, 2 + 2 + 4 + 16),
    GGMLQuantizationType.Q8_0:    (32, 2 + 32),
    GGMLQuantizationType.Q8_1:    (32, 4 + 4 + 32),
    GGMLQuantizationType.Q2_K:    (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLQuantizationType.Q3_K:    (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    GGMLQuantizationType.Q4_K:    (256, 2 + 2 + QK_K // 2 + 12),
    GGMLQuantizationType.Q5_K:    (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    GGMLQuantizationType.Q6_K:    (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.Q8_K:    (256, 4 + QK_K + QK_K // 8),
    GGMLQuantizationType.IQ2_XXS: (256, 2 + QK_K // 4),
    GGMLQuantizationType.IQ2_XS:  (256, 2 + QK_K // 4 + QK_K // 32),
    GGMLQuantizationType.IQ3_XXS: (256, 2 + QK_K // 4 + QK_K // 8),
    GGMLQuantizationType.IQ1_S:   (256, 2 + QK_K // 8 + QK_K // 16),
    GGMLQuantizationType.IQ4_NL:  (32, 2 + 16),
    GGMLQuantizationType.IQ3_S:   (256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    GGMLQuantizationType.IQ2_S:   (256, 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.IQ4_XS:  (256, 2 + 2 + QK_K // 2 + QK_K // 64),
    GGMLQuantizationType.I8:      (1, 1),
    GGMLQuantizationType.I16:     (1, 2),
    GGMLQuantizationType.I32:     (1, 4),
    GGMLQuantizationType.I64:     (1, 8),
    GGMLQuantizationType.F64:     (1, 8),
    GGMLQuantizationType.IQ1_M:   (256, QK_K // 8 + QK_K // 16  + QK_K // 32),
    GGMLQuantizationType.BF16:    (1, 2),
    GGMLQuantizationType.TQ1_0:   (256, 2 + 4 * 13),
    GGMLQuantizationType.TQ2_0:   (256, 2 + 64),
    GGMLQuantizationType.MXFP4:   (32, 1 + 16),
    GGMLQuantizationType.NVFP4:   (64, 4 + 32),
    GGMLQuantizationType.Q1_0:    (128, 2 + 16),
    GGMLQuantizationType.Q2_0:    (64, 2 + 16),
}
