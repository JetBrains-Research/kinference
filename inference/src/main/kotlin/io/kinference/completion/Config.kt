package io.kinference.completion

data class TokenizerConfig(val vocabPath: String, val mergesPath: String, val maxSeqLen: Int)

data class ModelConfig(val modelPath: String, val numAttentionHeads: Int, val hiddenSize: Int, val numLayer: Int, val vocabSize: Int)

data class GenerationConfig(val minLen: Int, val maxLen: Int, val numBeams: Int, val numGroups: Int,
                            val repetitionPenalty: Double, val lengthPenalty: Double,
                            val lenNormBase: Double, val lenNormPow: Double,
                            val prefixErrLimit: Int, val spellProb: Double)

data class FilterConfig(val minSymbolLen: Int, val minAvgLogProb: Double, val minProb: Double)

data class Config(val numSeqs: Int, val tokenizer: TokenizerConfig, val model: ModelConfig,
                val generation: GenerationConfig, val filter: FilterConfig)


val bpeTokenizer = TokenizerConfig(
    "/Users/aleksandr.khvorov/.cache/torch/transformers/71cc2431cf0b5bbe7a23601a808ed322c90251c8261b46f04970140a3c2c1cb4.1512018be4ba4e8726e41b9145129dc30651ea4fec86aa61f4b9f40bf94eac71",
    "/Users/aleksandr.khvorov/.cache/torch/transformers/4faf7afb02a1ea7d2944e9ba7a175c7b8de4957cdbae75cd5ddffc7c7643ebbc.70bec105b4158ed9a1747fea67a43f5dee97855c64d62b6ec3742f4cfdb5feda",
    1024
)

val model27 = ModelConfig(
    "/Users/aleksandr.khvorov/jb/grazie/grazie-datasets/src/completion/big/opt/onnxrt/onnx_models/distilgpt2_l3_h12_d256_int8.onnx",
4, 256, 3, 50257
)

val defaultGenerationConfig = GenerationConfig(
    1, 3, 5, 1,
1.0, 1.0,
5.0, 0.7,
0, 0.0001
)

val filterConfig = FilterConfig(2, -100.0, 0.0)

val config = Config(10, bpeTokenizer, model27, defaultGenerationConfig, filterConfig)
