package io.kinference.ndarray.extensions.reverse

sealed class ReverseSeqMode {
    abstract fun index(batchIdx: Int, seqIdx: Int, numBatches: Int, maxSeqLength: Int, blockSize: Int): Int
    abstract fun reverseIndex(batchIdx: Int, seqIdx: Int, seqLen: Int, numBatches: Int, maxSeqLength: Int, blockSize: Int): Int

    data object BatchMajorMode : ReverseSeqMode() {
        override fun index(batchIdx: Int, seqIdx: Int, numBatches: Int, maxSeqLength: Int, blockSize: Int): Int {
            return batchIdx * maxSeqLength * blockSize + seqIdx * blockSize
        }

        override fun reverseIndex(batchIdx: Int, seqIdx: Int, seqLen: Int, numBatches: Int, maxSeqLength: Int, blockSize: Int): Int {
            return batchIdx * maxSeqLength * blockSize + (seqLen - seqIdx - 1) * blockSize
        }
    }

    data object TimeMajorMode : ReverseSeqMode() {
        override fun index(batchIdx: Int, seqIdx: Int, numBatches: Int, maxSeqLength: Int, blockSize: Int): Int {
            return seqIdx * numBatches * blockSize + batchIdx * blockSize
        }

        override fun reverseIndex(batchIdx: Int, seqIdx: Int, seqLen: Int, numBatches: Int, maxSeqLength: Int, blockSize: Int): Int {
            return (seqLen - seqIdx - 1) * numBatches * blockSize + batchIdx * blockSize
        }
    }

    companion object {
        fun get(batchAxis: Int, timeAxis: Int): ReverseSeqMode {
            require(batchAxis in 0..1) { "Batch axis must be either 0 or 1" }
            require(timeAxis in 0..1) { "Time axis must be either 0 or 1" }
            require(batchAxis != timeAxis) { "Batch and time axis must have different values, but both are $batchAxis" }

            return when {
                batchAxis == 1 && timeAxis == 0 -> TimeMajorMode
                batchAxis == 0 && timeAxis == 1 -> BatchMajorMode
                else -> error("This branch should be unreachable")
            }
        }
    }
}
