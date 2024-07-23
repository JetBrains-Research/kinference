package io.kinference.ndarray.arrays.memory

import kotlinx.atomicfu.AtomicLong
import kotlinx.atomicfu.atomic

class MemoryLimiter(private val memoryLimit: Long) {
    private var usedMemory: AtomicLong = atomic(0L)

    fun checkMemoryLimitAndAdd(returned: Long): Boolean {
        val currentMemory = usedMemory.addAndGet(returned)
        return if (currentMemory > memoryLimit) {
            usedMemory.addAndGet(-returned)
            false
        } else true
    }

    fun freeMemory(deducted: Long) {
        usedMemory.addAndGet(-deducted)
    }
}
