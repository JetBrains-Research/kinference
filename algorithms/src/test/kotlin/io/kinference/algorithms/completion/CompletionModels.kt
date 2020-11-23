package io.kinference.algorithms.completion

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import io.kinference.algorithms.completion.config.ModelConfig
import io.kinference.algorithms.completion.config.TokenizerConfig
import io.kinference.algorithms.completion.loader.ModelLoader
import io.kinference.loaders.S3Client
import java.io.File

object CompletionModels {
    private val testData = File("../build/test-data")

    private const val myV4Name = "/gpt2/grazie/distilled/quantized/v4/"
    private const val myV5Name = "/gpt2/grazie/distilled/quantized/v5/"

    data class CompletionConfig(val tokenizer: TokenizerConfig, val model: ModelConfig, val loader: ModelLoader)

    val v4: CompletionConfig by lazy { loadConfigs(myV4Name, "tests${myV4Name}") }
    val v5: CompletionConfig by lazy { loadConfigs(myV5Name, "tests${myV5Name}") }


    private fun loadConfigs(name: String, prefix: String): CompletionConfig {
        val toFolder = File(testData, name)
        S3Client.copyObjects(prefix, toFolder)

        return CompletionConfig(getTokenizerConfig(toFolder), getModelConfig(toFolder), getModelLoader(toFolder))
    }

    private fun getConfig(toFolder: File): Map<String, Int> {
        val configPath = File(toFolder, "config.json").absolutePath
        val mapType = object : TypeReference<HashMap<String, Int>>() {}
        return ObjectMapper().readValue(File(configPath), mapType)
    }

    private fun getTokenizerConfig(toFolder: File): TokenizerConfig {
        val configJson = getConfig(toFolder)

        val maxSeqLen = configJson.getOrDefault("n_ctx", 30)

        return TokenizerConfig(maxSeqLen)
    }

    private fun getModelLoader(toFolder: File): ModelLoader {
        val modelPath = File(toFolder, "model.onnx")
        val vocabPath = File(toFolder, "vocab.json")
        val mergesPath = File(toFolder, "merges.txt")

        return ModelLoader.FileModelLoader(modelPath, vocabPath, mergesPath)
    }

    private fun getModelConfig(toFolder: File): ModelConfig {
        val configJson = getConfig(toFolder)

        val numAttentionHeads = configJson.getOrDefault("n_head", 8)
        val hiddenSize = configJson.getOrDefault("n_emb", 256)
        val numLayer = configJson.getOrDefault("n_layer", 4)
        val vocabSize = configJson.getOrDefault("vocab_size", 20000)

        return ModelConfig(numAttentionHeads, hiddenSize, numLayer, vocabSize)
    }
}
