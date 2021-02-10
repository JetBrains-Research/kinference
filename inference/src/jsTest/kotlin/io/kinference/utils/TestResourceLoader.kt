package io.kinference.utils

import kotlinx.browser.window
import kotlinx.coroutines.await
import org.khronos.webgl.Int8Array

actual object TestResourceLoader : ResourceLoader {
    override suspend fun fileBytes(path: String): ByteArray {
        val response = window.fetch(path).await()
        val buffer = response.arrayBuffer().await()
        val bytes = Int8Array(buffer).unsafeCast<ByteArray>()

        return bytes
    }

    override suspend fun fileText(path: String): String {
        val response = window.fetch(path).await()
        val text = response.text().await()
        return text
    }
}
