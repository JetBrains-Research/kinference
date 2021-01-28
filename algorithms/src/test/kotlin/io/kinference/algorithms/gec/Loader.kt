package io.kinference.algorithms.gec

import io.kinference.loaders.S3Client
import java.io.File

object ConfigLoader {
    private val testData = File("../build/test-data")

    private const val V2Path = "/bert/gec/en/standard/v2/"

    val v2: GECConfig by lazy { loadConfigs(V2Path, "tests$V2Path") }

    private fun loadConfigs(path: String, prefix: String): GECConfig {
        val toFolder = File(testData, path)
        S3Client.copyObjects(prefix, toFolder)

        return GECConfig.loadFrom { name -> File(toFolder, name).inputStream() }
    }
}

