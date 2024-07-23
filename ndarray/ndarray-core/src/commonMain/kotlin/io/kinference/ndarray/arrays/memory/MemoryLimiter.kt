package io.kinference.ndarray.arrays.memory

import kotlinx.atomicfu.AtomicInt
import kotlinx.atomicfu.atomic

interface MemoryLimiter {
    fun checkMemoryLimitAndAdd(returned: Int): Boolean
    fun freeMemory(deducted: Int)
}

object UnlimitedMemoryLimiter : MemoryLimiter {
    override fun checkMemoryLimitAndAdd(returned: Int): Boolean {
        return true
    }

    override fun freeMemory(deducted: Int) {

    }
}

class LimitedMemoryLimiter(val memoryLimit: Int) : MemoryLimiter {
    private var usedMemory: AtomicInt = atomic(0)

    override fun checkMemoryLimitAndAdd(returned: Int): Boolean {
        val currentMemory = usedMemory.addAndGet(returned)
        return if (currentMemory > memoryLimit) {
            usedMemory.addAndGet(-returned)
            false
        } else true
    }

    override fun freeMemory(deducted: Int) {
        usedMemory.addAndGet(-deducted)
    }
}
