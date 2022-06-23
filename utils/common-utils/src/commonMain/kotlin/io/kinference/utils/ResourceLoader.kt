package io.kinference.utils

import okio.Path

interface DataLoader {
    suspend fun bytes(path: Path): ByteArray
    suspend fun text(path: Path): String
}

expect object CommonDataLoader : DataLoader {
    override suspend fun bytes(path: Path): ByteArray
    override suspend fun text(path: Path): String
}
