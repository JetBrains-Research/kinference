package io.kinference.examples

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
