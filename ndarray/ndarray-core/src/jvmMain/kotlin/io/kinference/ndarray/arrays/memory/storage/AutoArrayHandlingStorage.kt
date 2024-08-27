package io.kinference.ndarray.arrays.memory.storage

import io.kinference.ndarray.arrays.memory.*
import io.kinference.primitives.types.DataType

internal interface TypedAutoHandlingStorage {
    fun getBlock(blocksNum: Int, blockSize: Int, limiter: MemoryManager): Array<Any>
    fun moveBlocksIntoUnused()
}

internal class AutoArrayHandlingStorage(private val limiter: MemoryManager) : ArrayStorage {
    private val storage: List<TypedAutoHandlingStorage> = listOf(
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

    internal fun getArrays(type: DataType, size: Int, count: Int): Array<Any> {
        return storage[type.ordinal].getBlock(blocksNum = count, blockSize = size, limiter = limiter)
    }

    override fun resetState() {
        storage.forEach { it.moveBlocksIntoUnused() }
        limiter.resetLimit()
    }
}
