package io.kinference.utils

enum class AllocationMode {
    NoAllocation,
    Manual,
    Auto;
}

class PredictionConfig private constructor(
    val parallelismLimit: Int,
    val allocationMode: AllocationMode,
    val memoryThreshold: Long,
    val memoryClearingInterval: Long
) {
    companion object {
        operator fun invoke(
            parallelismLimit: Int,
            allocationMode: AllocationMode,
            memoryThreshold: Long,
            memoryClearingInterval: Long
        ): PredictionConfig {
            require(parallelismLimit in 1..PlatformUtils.cores) {
                "Parallelism limit must be within the range of 1 to available CPU cores: ${PlatformUtils.cores}."
            }
            return if (allocationMode == AllocationMode.NoAllocation) {
                PredictionConfig(parallelismLimit, allocationMode, 0L, Long.MAX_VALUE)
            } else {
                require(memoryThreshold > 0) {
                    "Memory threshold must be positive."
                }
                require(memoryClearingInterval > 0) {
                    "Memory clearing interval must be positive."
                }

                PredictionConfig(parallelismLimit, allocationMode, memoryThreshold, memoryClearingInterval)
            }
        }
    }
}

object PredictionConfigs {
    val DefaultAutoAllocator: PredictionConfig = PredictionConfig(
        parallelismLimit = PlatformUtils.cores,
        allocationMode = AllocationMode.Auto,
        memoryThreshold = (PlatformUtils.maxHeap * 0.3).toLong(),
        memoryClearingInterval = 500
    )
    val DefaultManualAllocator: PredictionConfig = PredictionConfig(
        parallelismLimit = PlatformUtils.cores,
        allocationMode = AllocationMode.Manual,
        memoryThreshold = 50 * 1024 * 1024,
        memoryClearingInterval = 500
    )
    val NoAllocator: PredictionConfig = PredictionConfig(
        parallelismLimit = PlatformUtils.cores,
        allocationMode = AllocationMode.NoAllocation,
        memoryThreshold = 0L,
        memoryClearingInterval = Long.MAX_VALUE
    )

    fun customPredictionConfig(parallelismLimit: Int,
                               allocationMode: AllocationMode,
                               memoryThreshold: Long,
                               memoryClearingInterval: Long): PredictionConfig {
        return PredictionConfig(parallelismLimit, allocationMode, memoryThreshold, memoryClearingInterval)
    }
}
