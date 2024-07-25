package io.kinference.ndarray.arrays.memory

import io.kinference.utils.PlatformUtils
import kotlinx.atomicfu.AtomicLong
import kotlinx.atomicfu.atomic

interface MemoryLimiter {
    fun checkMemoryLimitAndAdd(returned: Long): Boolean
    fun freeMemory(deducted: Long)
}

class BaseMemoryLimiter(private val memoryLimit: Long) : MemoryLimiter {
    private var usedMemory: AtomicLong = atomic(0L)

    override fun checkMemoryLimitAndAdd(returned: Long): Boolean {
        val currentMemory = usedMemory.addAndGet(returned)
        return if (currentMemory > memoryLimit) {
            usedMemory.addAndGet(-returned)
            false
        } else true
    }

    override fun freeMemory(deducted: Long) {
        usedMemory.addAndGet(-deducted)
    }
}

object MemoryLimiters {
    val Default: MemoryLimiter = BaseMemoryLimiter((PlatformUtils.maxHeap * 0.3).toLong())
    val NoAllocator: MemoryLimiter = NoAllocatorMemoryLimiter
}

internal object NoAllocatorMemoryLimiter : MemoryLimiter {
    override fun checkMemoryLimitAndAdd(returned: Long): Boolean {
        return false
    }

    override fun freeMemory(deducted: Long) {

    }
}
