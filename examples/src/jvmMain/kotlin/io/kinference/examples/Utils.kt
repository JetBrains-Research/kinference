package io.kinference.examples

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.ndarray.arrays.NumberNDArrayCore
import io.ktor.client.HttpClient
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.request.prepareRequest
import io.ktor.client.statement.bodyAsChannel
import io.ktor.util.cio.writeChannel
import io.ktor.utils.io.copyAndClose
import java.io.File

val resourcesPath = System.getProperty("user.dir") + "/cache/"

/**
 * Downloads a file from the specified URL and saves it to the given output path.
 * If the file already exists at the output path, the download is skipped.
 *
 * @param url The URL from which the file will be downloaded.
 * @param outputPath The path to which the downloaded file will be saved.
 */
suspend fun downloadFile(url: String, outputPath: String) {
    // Check if the file already exists
    val file = File(outputPath)
    if (file.exists()) {
        println("File already exists at $outputPath. Skipping download.")
        return // Exit the function if the file exists
    }

    // Create an instance of HttpClient with custom timeout settings
    val client = HttpClient {
        install(HttpTimeout) {
            requestTimeoutMillis = 600_000 // Set timeout to 10 minutes (600,000 milliseconds)
        }
    }

    // Download the file and write to the specified output path
    client.prepareRequest(url).execute { response ->
        response.bodyAsChannel().copyAndClose(File(outputPath).writeChannel())
    }

    client.close()
}

suspend fun extractTopToken(output: Map<String, KIONNXData<*>>, tokensSize: Int, outputName: String): Long {
    val logits = output[outputName]!! as KITensor
    val sliced = logits.data.slice(
        starts = intArrayOf(0, 0, tokensSize - 1, 0),   // First batch, first element in the second dimension, last token, first vocab entry
        ends = intArrayOf(1, 1, tokensSize, 50257),     // Same batch, same second dimension, one token step, whole vocab (50257)
        steps = intArrayOf(1, 1, 1, 1)                  // Step of 1 for each dimension
    ) as NumberNDArrayCore
    val softmax = sliced.softmax(axis = -1)
    val topK = softmax.topK(
        axis = -1,                                      // Apply top-k along the last dimension (vocabulary size)
        k = 1,                                          // Retrieve the top 1 element
        largest = true,                                 // We want the largest probabilities (most probable tokens)
        sorted = false                                  // Sorting is unnecessary since we are only retrieving the top 1
    )
    val tokenId = (topK.second as LongNDArray)[intArrayOf(0, 0, 0, 0)]

    return tokenId
}
