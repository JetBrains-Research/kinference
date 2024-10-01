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

/**
 * Directory used to store cached files.
 *
 * This variable combines the user's current working directory
 * with a "cache" subdirectory to create the path for storing cache files.
 * It is used in various functions to check for existing files or directories,
 * create new ones if they do not exist, and manage the caching of downloaded files.
 */
val cacheDirectory = System.getProperty("user.dir") + "/.cache/"

/**
 * Downloads a file from the given URL and saves it with the specified file name.
 *
 * Checks if the directory specified by `cacheDirectory` exists.
 * If not, it creates the directory. If the file already exists,
 * the download is skipped. Otherwise, the file is downloaded
 * using an HTTP client with a 10-minute timeout setting.
 *
 * @param url The URL from which to download the file.
 * @param fileName The name to use for the downloaded file.
 * @param timeout Optional timeout duration for the download request, in milliseconds.
 * Defaults to 600,000 milliseconds (10 minutes).
 * Increase the timeout if you are not sure that download for the particular model with fit into the default timeout.
 */
suspend fun downloadFile(url: String, fileName: String, timeout: Long = 600_000) {
    // Ensure the predefined path is treated as a directory
    val directory = File(cacheDirectory)

    // Check if the directory exists, if not create it
    if (!directory.exists()) {
        println("Predefined directory doesn't exist. Creating directory at $cacheDirectory.")
        directory.mkdirs() // Create the directory if it doesn't exist
    }

    // Check if the file already exists
    val file = File(directory, fileName)
    if (file.exists()) {
        println("File already exists at ${file.absolutePath}. Skipping download.")
        return // Exit the function if the file exists
    }

    // Create an instance of HttpClient with custom timeout settings
    val client = HttpClient {
        install(HttpTimeout) {
            requestTimeoutMillis = timeout
        }
    }

    // Download the file and write to the specified output path
    client.prepareRequest(url).execute { response ->
        response.bodyAsChannel().copyAndClose(file.writeChannel())
    }

    client.close()
}

/**
 * Extracts the token ID with the highest probability from the output tensor.
 *
 * @param output A map containing the output tensors identified by their names.
 * @param tokensSize The number of tokens in the sequence.
 * @param outputName The name of the tensor containing the logits.
 * @return The ID of the top token.
 */
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
