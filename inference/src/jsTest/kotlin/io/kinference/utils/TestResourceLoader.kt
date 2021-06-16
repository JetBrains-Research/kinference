package io.kinference.utils

import kotlinx.browser.window
import kotlinx.coroutines.await
import org.khronos.webgl.Int8Array

actual object ResourcesTestDataLoader : TestDataLoader {
    actual override suspend fun bytes(path: TestDataLoader.Path): ByteArray {
        val response = window.fetch(path.toRelativePath()).await()
        val buffer = response.arrayBuffer().await()

        return Int8Array(buffer).unsafeCast<ByteArray>()
    }

    actual override suspend fun text(path: TestDataLoader.Path): String {
        val response = window.fetch(path.toRelativePath()).await()
        return response.text().await()
    }
}

actual object S3TestDataLoader : TestDataLoader {
    actual override suspend fun bytes(path: TestDataLoader.Path): ByteArray {
        val response = window.fetch(TestDataLoader.Path("s3", path).toRelativePath()).await()
        val buffer = response.arrayBuffer().await()

        return Int8Array(buffer).unsafeCast<ByteArray>()
    }

    actual override suspend fun text(path: TestDataLoader.Path): String {
        val response = window.fetch(TestDataLoader.Path("s3", path).toRelativePath()).await()
        return response.text().await()
    }
}

