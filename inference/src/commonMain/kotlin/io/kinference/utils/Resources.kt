package io.kinference.utils

interface ResourceLoader {
    suspend fun fileBytes(path: String): ByteArray
    suspend fun fileText(path: String): String
}
