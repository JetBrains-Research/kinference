package io.kinference.ndarray.arrays.memory

import io.kinference.primitives.types.DataType
import io.kinference.utils.PlatformUtils
import kotlinx.atomicfu.*

interface MemoryLimiter {
    /**
     * Checks if the memory limit allows adding the specified amount of memory and performs the addition
     *
     * @param type is the DataType of underlying primitives in a checking array
     * @param size is the checking array size
     * @return true if the memory was added successfully and false if adding the memory exceeds the memory limit
     */
    fun checkMemoryLimitAndAdd(type: DataType, size: Int): Boolean

    /**
     * Resets the used memory into 0L
     */
    fun resetLimit()
}

class BaseMemoryLimiter internal constructor(private val memoryLimit: Long) : MemoryLimiter {
    private var usedMemory: AtomicLong = atomic(0L)

    override fun checkMemoryLimitAndAdd(type: DataType, size: Int): Boolean {
        // Attempt to add memory and check the limit
        val added = sizeInBytes(type.ordinal, size)
        val successful = usedMemory.getAndUpdate { current ->
            if (current + added > memoryLimit) current else current + added
        } != usedMemory.value // Check if the update was successful

        return successful
    }

    override fun resetLimit() {
        usedMemory.value = 0L
    }

    companion object {
        private val typeSizes: LongArray = LongArray(DataType.entries.size).apply {
            this[DataType.BYTE.ordinal] = Byte.SIZE_BYTES.toLong()
            this[DataType.SHORT.ordinal] = Short.SIZE_BYTES.toLong()
            this[DataType.INT.ordinal] = Int.SIZE_BYTES.toLong()
            this[DataType.LONG.ordinal] = Long.SIZE_BYTES.toLong()

            this[DataType.UBYTE.ordinal] = UByte.SIZE_BYTES.toLong()
            this[DataType.USHORT.ordinal] = UShort.SIZE_BYTES.toLong()
            this[DataType.UINT.ordinal] = UInt.SIZE_BYTES.toLong()
            this[DataType.ULONG.ordinal] = ULong.SIZE_BYTES.toLong()

            this[DataType.FLOAT.ordinal] = Float.SIZE_BYTES.toLong()
            this[DataType.DOUBLE.ordinal] = Double.SIZE_BYTES.toLong()

            this[DataType.BOOLEAN.ordinal] = 1.toLong()
        }

        private fun sizeInBytes(typeIndex: Int, size: Int): Long {
            return typeSizes[typeIndex] * size
        }
    }
}

object MemoryLimiters {
    val DefaultAutoAllocator: MemoryLimiter = BaseMemoryLimiter((PlatformUtils.maxHeap * 0.3).toLong())
    val DefaultManualAllocator: MemoryLimiter = BaseMemoryLimiter(50 * 1024 * 1024)
    val NoAllocator: MemoryLimiter = BaseMemoryLimiter(0L)

    fun customLimiter(memoryLimit: Long): MemoryLimiter {
        return BaseMemoryLimiter(memoryLimit)
    }
}
