package io.kinference.utils

import java.io.File


actual object ResourcesTestDataLoader : TestDataLoader {
    private val file = File("../../utils/test-utils")

    actual override suspend fun bytes(path: TestDataLoader.Path): ByteArray {
        return File(file, path.toRelativePath()).readBytes()
    }

    actual override suspend fun text(path: TestDataLoader.Path): String {
        return File(file, path.toRelativePath()).readText()
    }
}

actual object S3TestDataLoader : TestDataLoader {
    private val file = File("../../test-data/s3/tests")

    actual override suspend fun bytes(path: TestDataLoader.Path): ByteArray = File(file, path.toRelativePath()).readBytes()

    actual override suspend fun text(path: TestDataLoader.Path): String = File(file, path.toRelativePath()).readText()
}
