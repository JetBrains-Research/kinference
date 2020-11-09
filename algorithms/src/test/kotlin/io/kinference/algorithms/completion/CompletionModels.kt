package io.kinference.algorithms.completion

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import io.kinference.loaders.S3Client
import java.io.File

object CompletionModels {
    private val testData = File("../build/test-data")

    private const val myV4Name = "/gpt/grazie/distilled/quantized/v4/"
    private const val myV5Name = "/gpt/grazie/distilled/quantized/v5/"

    data class CompletionConfig(val tokenizer: TokenizerConfig, val model: ModelConfig)

    val v4: CompletionConfig by lazy { loadConfigs(myV4Name, "tests/${myV4Name}") }
    val v5: CompletionConfig by lazy { loadConfigs(myV5Name, "tests/${myV5Name}") }


    object Config {
        val generation = GenerationConfig(
            1, 3, 5, 1,
            1.0, 1.0,
            5.0, 0.7,
            0, 0.0001
        )

        val filter = FilterConfig(2, -100.0, 0.0)
    }

    private fun loadConfigs(name: String, prefix: String): CompletionConfig {
        val toFolder = File(testData, name)
        S3Client.copyObjects(prefix, toFolder)

        return CompletionConfig(getTokenizerConfig(toFolder), getModelConfig(toFolder))
    }

    private fun getConfig(toFolder: File): Map<String, Int> {
        val configPath = File(toFolder, "config.json").absolutePath
        val mapType = object : TypeReference<HashMap<String, Int>>() {}
        return ObjectMapper().readValue(File(configPath), mapType)
    }

    private fun getTokenizerConfig(toFolder: File): TokenizerConfig {
        val configJson = getConfig(toFolder)

        val vocabPath = File(toFolder, "vocab.json").absolutePath
        val mergesPath = File(toFolder, "merges.txt").absolutePath

        val maxSeqLen = configJson.getOrDefault("n_ctx", 30)

        return TokenizerConfig(vocabPath, mergesPath, maxSeqLen)
    }

    private fun getModelConfig(toFolder: File): ModelConfig {
        val configJson = getConfig(toFolder)

        val modelPath = File(toFolder, "model.onnx").absolutePath

        val numAttentionHeads = configJson.getOrDefault("n_head", 8)
        val hiddenSize = configJson.getOrDefault("n_emb", 256)
        val numLayer = configJson.getOrDefault("n_layer", 4)
        val vocabSize = configJson.getOrDefault("vocab_size", 20000)

        return ModelConfig(modelPath, numAttentionHeads, hiddenSize, numLayer, vocabSize)
    }
}
