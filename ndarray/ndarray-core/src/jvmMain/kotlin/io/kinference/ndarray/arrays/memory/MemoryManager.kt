package io.kinference.ndarray.arrays.memory

import io.kinference.primitives.types.DataType
import kotlinx.atomicfu.*
import kotlinx.coroutines.*

internal class MemoryManager internal constructor(private val memoryLimit: Long, private val cacheClearingInterval: Long, private val onCacheClear: () -> Unit) {
    private var usedMemory: AtomicLong = atomic(0L)
    private val lastAccessTime = atomic(System.currentTimeMillis())
    private val monitorJob: AtomicRef<Job?> = atomic(initial = null)
    private val isFinalized = atomic(initial = false)

    /**
     * Checks if the memory limit allows adding the specified amount of memory and performs the addition
     *
     * @param type is the DataType of underlying primitives in a checking array
     * @param size is the checking array size
     * @return true if the memory was added successfully and false if adding the memory exceeds the memory limit
     */
    fun checkMemoryLimitAndAdd(type: DataType, size: Int): Boolean {
        // Attempt to add memory and check the limit
        val added = sizeInBytes(type.ordinal, size)
        val successful = usedMemory.getAndUpdate { current ->
            if (current + added > memoryLimit) current else current + added
        } != usedMemory.value // Check if the update was successful

        return successful
    }

    /**
     * Resets the used memory into 0L
     */
    fun resetLimit() {
        usedMemory.value = 0L
    }

    fun updateLastAccessTime() {
        lastAccessTime.value = System.currentTimeMillis()

        // Start monitoring if not already started
        if (monitorJob.compareAndSet(expect = null, update = null) && !isFinalized.value) {
            val newJob = CoroutineScope(Dispatchers.Default).launch {
                while (isActive) {
                    delay(cacheClearingInterval)
                    if (System.currentTimeMillis() - lastAccessTime.value > cacheClearingInterval) {
                        onCacheClear()
                    }
                }
            }
            if (!monitorJob.compareAndSet(expect = null, newJob)) {
                newJob.cancel() // Cancel if another thread set the job
            }
        }
    }

    fun stopMonitoring() {
        if (isFinalized.compareAndSet(expect = false, update = true)) {
            monitorJob.getAndSet(value = null)?.cancel()
        }
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
