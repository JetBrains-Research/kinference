package io.kinference.algorithms.completion.config

import io.kinference.algorithms.completion.loader.ModelLoader


data class Config(
    val loader: ModelLoader,
    val numSeqs: Int,
    val tokenizer: TokenizerConfig,
    val model: ModelConfig,
    val generation: GenerationConfig,
    val filter: FilterConfig
)
