package io.kinference.ndarray.arrays.memory

import io.kinference.utils.PlatformUtils
import kotlinx.atomicfu.*

interface MemoryLimiter {
    /**
     * Checks if the memory limit allows adding the specified amount of memory and performs the addition
     *
     * @param added the memory in bytes to add
     * @return true if the memory was added successfully and false if adding the memory exceeds the memory limit
     */
    fun checkMemoryLimitAndAdd(added: Long): Boolean

    /**
     * Resets the used memory into 0L
     */
    fun resetLimit()
}

class BaseMemoryLimiter internal constructor(private val memoryLimit: Long) : MemoryLimiter {
    private var usedMemory: AtomicLong = atomic(0L)

    override fun checkMemoryLimitAndAdd(added: Long): Boolean {
        // Attempt to add memory and check the limit
        val successful = usedMemory.getAndUpdate { current ->
            if (current + added > memoryLimit) current else current + added
        } != usedMemory.value // Check if the update was successful

        return successful
    }

    override fun resetLimit() {
        usedMemory.value = 0L
    }
}

object MemoryLimiters {
    val Default: MemoryLimiter = BaseMemoryLimiter((PlatformUtils.maxHeap * 0.3).toLong())
    val NoAllocator: MemoryLimiter = BaseMemoryLimiter(0L)

    fun customLimiter(memoryLimit: Long): MemoryLimiter {
        return BaseMemoryLimiter(memoryLimit)
    }
}
