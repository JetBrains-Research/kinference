package io.kinference.ndarray.arrays.memory.storage

import io.kinference.ndarray.arrays.memory.*

internal interface TypedAutoHandlingStorage {
    fun moveBlocksIntoUnused()
}

internal class AutoArrayHandlingStorage(internal val limiter: MemoryManager) : ArrayStorage {
    internal val storage: List<TypedAutoHandlingStorage> = listOf(
        ByteAutoHandlingArrayStorage(),
        ShortAutoHandlingArrayStorage(),
        IntAutoHandlingArrayStorage(),
        LongAutoHandlingArrayStorage(),
        UByteAutoHandlingArrayStorage(),
        UShortAutoHandlingArrayStorage(),
        UIntAutoHandlingArrayStorage(),
        ULongAutoHandlingArrayStorage(),
        FloatAutoHandlingArrayStorage(),
        DoubleAutoHandlingArrayStorage(),
        BooleanAutoHandlingArrayStorage()
    )

    override fun resetState() {
        storage.forEach { it.moveBlocksIntoUnused() }
        limiter.resetLimit()
    }
}
