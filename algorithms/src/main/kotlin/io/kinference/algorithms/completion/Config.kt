package io.kinference.algorithms.completion

data class TokenizerConfig(val vocabPath: String, val mergesPath: String, val maxSeqLen: Int)

data class ModelConfig(val modelPath: String, val numAttentionHeads: Int, val hiddenSize: Int, val numLayer: Int, val vocabSize: Int)

data class GenerationConfig(val minLen: Int, val maxLen: Int, val numBeams: Int, val numGroups: Int,
                            val repetitionPenalty: Double, val lengthPenalty: Double,
                            val lenNormBase: Double, val lenNormPow: Double,
                            val prefixErrLimit: Int, val spellProb: Double)

data class FilterConfig(val minSymbolLen: Int, val minAvgLogProb: Double, val minProb: Double)

data class Config(val numSeqs: Int, val tokenizer: TokenizerConfig, val model: ModelConfig,
                val generation: GenerationConfig, val filter: FilterConfig)
