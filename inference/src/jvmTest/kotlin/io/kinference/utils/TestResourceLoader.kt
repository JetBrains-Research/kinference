package io.kinference.utils

import java.io.File

actual object TestResourceLoader : ResourceLoader {
    override suspend fun fileBytes(path: String): ByteArray = File(path).readBytes()
    override suspend fun fileText(path: String): String = File(path).readText()
}
