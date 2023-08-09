package io.kinference.ndarray.extensions.reverse

enum class ReverseSeqMode {
    BATCH_MAJOR,
    TIME_MAJOR;

    companion object {
        fun get(batchAxis: Int, timeAxis: Int): ReverseSeqMode {
            require(batchAxis in 0..1) { "Batch axis must be either 0 or 1" }
            require(timeAxis in 0..1) { "Time axis must be either 0 or 1" }
            require(batchAxis != timeAxis) { "Batch and time axis must have different values, but both are $batchAxis" }

            return when {
                batchAxis == 1 && timeAxis == 0 -> TIME_MAJOR
                batchAxis == 0 && timeAxis == 1 -> BATCH_MAJOR
                else -> error("This branch should be unreachable")
            }
        }
    }
}
