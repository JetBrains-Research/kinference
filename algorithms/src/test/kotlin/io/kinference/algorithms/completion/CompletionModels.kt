package io.kinference.algorithms.completion

import io.kinference.loaders.S3Client
import java.io.File

object CompletionModels {
    private val testData = File("../build/test-data")

    private const val myV4Name = "/gpt2/grazie/distilled/quantized/v4/"
    private const val myV5Name = "/gpt2/grazie/distilled/quantized/v5/"
    private const val myV6Name = "/gpt2/grazie/distilled/quantized/v6/"

    val v4: CompletionConfig by lazy { loadConfigs(myV4Name, "tests${myV4Name}") }
    val v5: CompletionConfig by lazy { loadConfigs(myV5Name, "tests${myV5Name}") }
    val v6: CompletionConfig by lazy { loadConfigs(myV5Name, "tests${myV6Name}") }


    private fun loadConfigs(name: String, prefix: String): CompletionConfig {
        val toFolder = File(testData, name)
        S3Client.copyObjects(prefix, toFolder)

        return CompletionConfig.fromFolder(10, toFolder)
    }
}
