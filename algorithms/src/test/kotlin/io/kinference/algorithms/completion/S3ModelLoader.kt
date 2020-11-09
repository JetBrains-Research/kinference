package io.kinference.algorithms.completion

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import java.io.File

object S3ModelLoader {
    private val testData = File("build/test-data")
    private val loadedModels = HashMap<String, File>()

    fun loadConfigs(testPath: String, prefix: String) {
        val toFolder = File(testData, testPath)
//        S3Client.copyObjects(prefix, toFolder)
        loadedModels[testPath] = toFolder
    }

    private fun getConfig(name: String): Map<String, Int> {
        val toFolder = loadedModels[name]
        val configPath = File(toFolder, "config.json").absolutePath
        val mapType = object : TypeReference<HashMap<String, Int>>() {}
        return ObjectMapper().readValue(File(configPath), mapType)
    }

    fun getTokenizerConfig(name: String): TokenizerConfig {
        val toFolder = loadedModels[name]
        val configJson = getConfig(name)

        val vocabPath = File(toFolder, "vocab.onnx").absolutePath
        val mergesPath = File(toFolder, "merges.txt").absolutePath

        val maxSeqLen = configJson.getOrDefault("n_ctx", 30)

        return TokenizerConfig(vocabPath, mergesPath, maxSeqLen)
    }

    fun getModelConfig(name: String): ModelConfig {
        val toFolder = loadedModels[name]
        val configJson = getConfig(name)

        val modelPath = File(toFolder, "model.onnx").absolutePath

        val numAttentionHeads = configJson.getOrDefault("n_head", 8)
        val hiddenSize = configJson.getOrDefault("n_emb", 256)
        val numLayer = configJson.getOrDefault("n_layer", 4)
        val vocabSize = configJson.getOrDefault("vocab_size", 20000)

        return ModelConfig(modelPath, numAttentionHeads, hiddenSize, numLayer, vocabSize)
    }
}
