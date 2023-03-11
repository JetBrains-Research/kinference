package io.kinference.utils

import okio.FileSystem
import okio.Path

actual object CommonDataLoader : DataLoader {
    actual override suspend fun bytes(path: Path): ByteArray {
        FileSystem.SYSTEM.read(path) {
            return readByteArray()
        }
    }

    actual override suspend fun text(path: Path): String {
        FileSystem.SYSTEM.read(path) {
            return readUtf8()
        }
    }
}
