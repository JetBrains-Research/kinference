package io.kinference.ndarray.arrays.memory

import io.kinference.utils.PlatformUtils
import kotlinx.atomicfu.AtomicLong
import kotlinx.atomicfu.atomic

interface MemoryLimiter {
    /**
     * Checks if the memory limit allows adding the specified amount of memory and performs the addition.
     *
     * @param added the memory in bytes to add
     * @return true if the memory was added successfully and false if adding the memory exceeds the memory limit
     */
    fun checkMemoryLimitAndAdd(added: Long): Boolean

    /**
     * Deducts the specified amount of memory from the memory limiter.
     *
     * @param deducted the memory in bytes to deduct from the memory limiter
     */
    fun deductMemory(deducted: Long)
}

class BaseMemoryLimiter(private val memoryLimit: Long) : MemoryLimiter {
    private var usedMemory: AtomicLong = atomic(0L)

    override fun checkMemoryLimitAndAdd(added: Long): Boolean {
        val currentMemory = usedMemory.addAndGet(added)
        return if (currentMemory > memoryLimit) {
            usedMemory.addAndGet(-added)
            false
        } else true
    }

    override fun deductMemory(deducted: Long) {
        usedMemory.addAndGet(-deducted)
    }
}

object MemoryLimiters {
    val Default: MemoryLimiter = BaseMemoryLimiter((PlatformUtils.maxHeap * 0.3).toLong())
    val NoAllocator: MemoryLimiter = NoAllocatorMemoryLimiter

    fun customLimiter(memoryLimit: Long): MemoryLimiter {
        return BaseMemoryLimiter(memoryLimit)
    }
}

internal object NoAllocatorMemoryLimiter : MemoryLimiter {
    override fun checkMemoryLimitAndAdd(added: Long): Boolean {
        return false
    }

    override fun deductMemory(deducted: Long) {

    }
}
