package io.kinference.ndarray.arrays.memory

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
     * @param sizeInBytes is the checking size of an array in bytes
     * @return true if the memory was added successfully and false if adding the memory exceeds the memory limit
     */
    fun checkMemoryLimitAndAdd(sizeInBytes: Long): Boolean {
        // Attempt to add memory and check the limit
        val successful = usedMemory.getAndUpdate { current ->
            if (current + sizeInBytes > memoryLimit) current else current + sizeInBytes
        } != usedMemory.value // Check if the update was successful

        return successful
    }

    /**
     * Resets the used memory into 0L
     */
    fun resetLimit() {
        usedMemory.value = 0L
    }

    /**
     * Updates the last access time to the current system time and starts a monitoring coroutine if it isn't already running.
     *
     * This function sets the `lastAccessTime` to the current system time in milliseconds.
     * It also initiates a monitoring coroutine to periodically check
     * if the time since the last access exceeds a predefined `cacheClearingInterval`.
     * If it does, the `onCacheClear` function is triggered to handle
     * any necessary cache clearing.
     * The coroutine will run only if it is not already running and `isFinalized` is false.
     */
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

    /**
     * Stops the monitoring process by canceling the active monitoring coroutine.
     *
     * This function sets the `isFinalized` flag to true, indicating that the monitoring process has been
     * concluded.
     * If a monitoring coroutine is currently active, it will be canceled.
     */
    fun stopMonitoring() {
        if (isFinalized.compareAndSet(expect = false, update = true)) {
            monitorJob.getAndSet(value = null)?.cancel()
        }
    }
}
