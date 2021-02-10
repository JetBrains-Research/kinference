package io.kinference.utils

import java.io.File


actual object ResourcesTestDataLoader : TestDataLoader {
    actual override suspend fun bytes(path: TestDataLoader.Path): ByteArray {
        return File(path.toRelativePath()).readBytes()
    }

    actual override suspend fun text(path: TestDataLoader.Path): String {
        return File(path.toRelativePath()).readText()
    }
}

actual object S3TestDataLoader : TestDataLoader {
    private val file = File("../build/s3/tests")

    actual override suspend fun bytes(path: TestDataLoader.Path): ByteArray = File(file, path.toRelativePath()).readBytes()

    actual override suspend fun text(path: TestDataLoader.Path): String = File(file, path.toRelativePath()).readText()
}
