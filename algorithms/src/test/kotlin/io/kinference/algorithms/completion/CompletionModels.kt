package io.kinference.algorithms.completion

import io.kinference.algorithms.completion.config.Config
import io.kinference.loaders.S3Client
import java.io.File

object CompletionModels {
    private val testData = File("../build/test-data")

    private const val myV4Name = "/gpt2/grazie/distilled/quantized/v4/"
    private const val myV5Name = "/gpt2/grazie/distilled/quantized/v5/"

    val v4: Config by lazy { loadConfigs(myV4Name, "tests${myV4Name}") }
    val v5: Config by lazy { loadConfigs(myV5Name, "tests${myV5Name}") }


    private fun loadConfigs(name: String, prefix: String): Config {
        val toFolder = File(testData, name)
        S3Client.copyObjects(prefix, toFolder)

        return Config.fromFolder(10, toFolder)
    }
}
