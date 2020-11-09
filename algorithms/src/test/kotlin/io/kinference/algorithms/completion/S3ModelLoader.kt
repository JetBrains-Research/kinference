package io.kinference.algorithms.completion

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import io.kinference.loaders.S3Client
import java.io.File

object S3ModelLoader {
    private val testData = File("../build/test-data")
    private val loadedModels = HashMap<String, File>()

    fun loadConfigs(name: String, prefix: String) : Pair<TokenizerConfig, ModelConfig> {
        val toFolder = File(testData, name)
        S3Client.copyObjects(prefix, toFolder)
        loadedModels[name] = toFolder

        return  getTokenizerConfig(toFolder) to getModelConfig(toFolder)
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
