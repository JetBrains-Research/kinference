package io.kinference.algorithms.completion

import io.kinference.algorithms.completion.loader.CompletionModelLoader
import java.io.File
import java.io.InputStream


/**
 * Configuration of a completion model
 *
 * @param loader is a loader with which model can be loaded
 * @param numSuggestions is a total number of suggestions model should generate
 * @param model is a configuration of GPT-like model, it is also should be taken
 * from `config.json` and is distributed with model
 * @param generation is a configuration of generation process
 * @param filter is a configuration of filters used during pre-filtering of completions
 */
data class CompletionConfig(
    val loader: CompletionModelLoader,
    val numSuggestions: Int,
    val model: Model,
    val generation: Generation,
    val filter: Filter
) {
    /**
     * Tokenizer configuration that is used in completion model
     */

    /**
     * GPT-like model configuration that is used in completion model
     */
    class Model(val numAttentionHeads: Int, val hiddenSize: Int, val numLayer: Int, val vocabSize: Int, val maxSeqLen: Int)

    /**
     * Configuration of pre-filtering used in completion model
     *
     * @param minSymbolLen is a minimum number of symbols that would be in completions, if
     * `completion.text.length < minSymbolLen` completion would be filtered out
     * @param minAvgLogProb is a minimum average log probability, if completion has it
     * smaller -- it would be filtered out
     * @param minProb is a minimum probability of completion, if completion has
     * it smaller -- it would be filtered out
     */
    data class Filter(val minSymbolLen: Int, val minAvgLogProb: Double, val minProb: Double) {
        companion object {
            /** Default filtering setup that can be used for common scenarios */
            val default = Filter(
                minSymbolLen = 2,
                minAvgLogProb = -100.0,
                minProb = 0.0
            )
        }
    }

    /**
     * Configuration of generation that is used in completion model
     *
     * @param minLen is a minimum length in words expected from completion model
     * @param maxLen is a maximum length in words expected from completion model
     */
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
            /** Default generation setup that can be used for common scenarios */
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
        /** Load completion model from specific folder and configure it to have at max [total] suggestions */
        fun fromFolder(total: Int, folder: File) = fromGetter(total) { File(folder, it).inputStream() }

        /**
         * Load completion model with [getter] and configure it to have at max [total] suggestions
         *
         * Note, that to [getter] would be passed the names of model artifacts, e.g. `model.onnx` or `config.json`
         */
        @Suppress("MemberVisibilityCanBePrivate")
        fun fromGetter(total: Int, getter: (String) -> InputStream): CompletionConfig {
            val loader = getModelLoader(getter)

            val model = getModelConfig(loader.getConfig())

            return CompletionConfig(loader, total, model, Generation.default, Filter.default)
        }


        private fun getModelLoader(getter: (String) -> InputStream): CompletionModelLoader = CompletionModelLoader.FromGetter(
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
            val maxSeqLen = config.getOrDefault("n_ctx", 30)

            return Model(numAttentionHeads, hiddenSize, numLayer, vocabSize, maxSeqLen)
        }
    }
}
