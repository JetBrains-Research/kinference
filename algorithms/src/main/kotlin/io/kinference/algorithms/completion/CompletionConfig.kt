package io.kinference.algorithms.completion

import io.kinference.algorithms.completion.loader.ModelLoader
import java.io.File
import java.io.InputStream


data class CompletionConfig(
    val loader: ModelLoader,
    val numSuggestions: Int,
    val tokenizer: Tokenizer,
    val model: Model,
    val generation: Generation,
    val filter: Filter
) {
    data class Tokenizer(val maxSeqLen: Int)

    class Model(val numAttentionHeads: Int, val hiddenSize: Int, val numLayer: Int, val vocabSize: Int)

    data class Filter(val minSymbolLen: Int, val minAvgLogProb: Double, val minProb: Double) {
        companion object {
            val default = Filter(
                minSymbolLen = 2,
                minAvgLogProb = -100.0,
                minProb = 0.0
            )
        }
    }

    data class Generation(
        val minLen: Int,
        val maxLen: Int,
        val numBeams: Int,
        val numGroups: Int,
        val repetitionPenalty: Double,
        val lengthPenalty: Double,
        val lenNormBase: Double,
        val lenNormPow: Double,
        val prefixErrLimit: Int,
        val spellProb: Double
    ) {

        companion object {
            val default = Generation(
                minLen = 1,
                maxLen = 3,
                numBeams = 5,
                numGroups = 1,
                repetitionPenalty = 1.0,
                lengthPenalty = 1.0,
                lenNormBase = 5.0,
                lenNormPow = 0.7,
                prefixErrLimit = 0,
                spellProb = 0.0001
            )
        }
    }


    companion object {
        fun fromFolder(total: Int, folder: File) = fromGetter(total) { File(folder, it).inputStream() }

        fun fromGetter(total: Int, getter: (String) -> InputStream): CompletionConfig {
            val loader = getModelLoader(getter)

            val tokenizer = getTokenizerConfig(loader.getConfig())
            val model = getModelConfig(loader.getConfig())

            return CompletionConfig(loader, total, tokenizer, model, Generation.default, Filter.default)
        }

        private fun getTokenizerConfig(config: Map<String, Int>): Tokenizer {
            val maxSeqLen = config.getOrDefault("n_ctx", 30)

            return Tokenizer(maxSeqLen)
        }

        private fun getModelLoader(getter: (String) -> InputStream): ModelLoader = ModelLoader.CustomModelLoader(
            { getter("model.onnx").use { it.readBytes() } },
            { getter("vocab.json").use { it.reader().readText() } },
            { getter("merges.txt").use { it.reader().readText() } },
            { getter("config.json").use { it.reader().readText() } }
        )

        private fun getModelConfig(config: Map<String, Int>): Model {
            val numAttentionHeads = config.getOrDefault("n_head", 8)
            val hiddenSize = config.getOrDefault("n_emb", 256)
            val numLayer = config.getOrDefault("n_layer", 4)
            val vocabSize = config.getOrDefault("vocab_size", 20000)

            return Model(numAttentionHeads, hiddenSize, numLayer, vocabSize)
        }
    }
}
