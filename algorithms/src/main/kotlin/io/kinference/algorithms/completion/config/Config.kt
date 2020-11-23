package io.kinference.algorithms.completion.config

import io.kinference.algorithms.completion.loader.ModelLoader
import java.io.File
import java.io.InputStream


data class Config(
    val loader: ModelLoader,
    val numSeqs: Int,
    val tokenizer: TokenizerConfig,
    val model: ModelConfig,
    val generation: GenerationConfig,
    val filter: FilterConfig
) {

    companion object {
        fun fromFolder(total: Int, folder: File) = fromGetter(total) { File(folder, it).inputStream() }

        fun fromGetter(total: Int, getter: (String) -> InputStream): Config {
            val loader = getModelLoader(getter)

            val tokenizer = getTokenizerConfig(loader.getConfig())
            val model = getModelConfig(loader.getConfig())

            return Config(loader, total, tokenizer, model, GenerationConfig.default, FilterConfig.default)
        }

        private fun getTokenizerConfig(config: Map<String, Int>): TokenizerConfig {
            val maxSeqLen = config.getOrDefault("n_ctx", 30)

            return TokenizerConfig(maxSeqLen)
        }

        private fun getModelLoader(getter: (String) -> InputStream): ModelLoader = ModelLoader.CustomModelLoader(
            { getter("model.onnx").use { it.readBytes() } },
            { getter("vocab.json").use { it.reader().readText() } },
            { getter("merges.txt").use { it.reader().readText() } },
            { getter("config.json").use { it.reader().readText() } }
        )

        private fun getModelConfig(config: Map<String, Int>): ModelConfig {
            val numAttentionHeads = config.getOrDefault("n_head", 8)
            val hiddenSize = config.getOrDefault("n_emb", 256)
            val numLayer = config.getOrDefault("n_layer", 4)
            val vocabSize = config.getOrDefault("vocab_size", 20000)

            return ModelConfig(numAttentionHeads, hiddenSize, numLayer, vocabSize)
        }
    }
}
