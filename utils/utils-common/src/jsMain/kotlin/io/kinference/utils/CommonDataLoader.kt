package io.kinference.utils

import kotlinx.browser.window
import kotlinx.coroutines.await
import okio.Path
import org.khronos.webgl.Int8Array

actual object CommonDataLoader : DataLoader {
    actual override suspend fun bytes(path: Path): ByteArray {
        val response = window.fetch(path.toString()).await()
        val buffer = response.arrayBuffer().await()

        return Int8Array(buffer).unsafeCast<ByteArray>()
    }

    actual override suspend fun text(path: Path): String {
        val response = window.fetch(path.toString()).await()
        return response.text().await()
    }
}
