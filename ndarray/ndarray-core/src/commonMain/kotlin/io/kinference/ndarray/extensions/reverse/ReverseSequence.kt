package io.kinference.ndarray.extensions.reverse

import io.kinference.ndarray.arrays.NDArrayCore
import io.kinference.ndarray.arrays.computeBlockSize
import io.kinference.ndarray.extensions.allocateNDArray

fun <T : NDArrayCore> T.reverseSeq(mode: ReverseSeqMode, seqLens: IntArray): T {
    val batchSize = if (mode == ReverseSeqMode.BATCH_MAJOR) shape[0] else shape[1]
    val maxSeqLen = if (mode == ReverseSeqMode.BATCH_MAJOR) shape[1] else shape[0]

    require(seqLens.size == batchSize) { "Sequence lengths array size must have $batchSize elements but the array of size ${seqLens.size} was found" }

    val blockSize = computeBlockSize(fromDim = 2)
    return when (mode) {
        ReverseSeqMode.BATCH_MAJOR -> reverseSequenceBatchMajor(seqLens, batchSize, maxSeqLen, blockSize)
        ReverseSeqMode.TIME_MAJOR -> reverseSequenceTimeMajor(seqLens, batchSize, maxSeqLen, blockSize)
    }
}

private fun <T : NDArrayCore> T.reverseSequenceBatchMajor(seqLens: IntArray, batchSize: Int, maxSeqLen: Int, blockSize: Int): T {
    val output = allocateNDArray(type, shape)

    for (batchIdx in 0 until batchSize) {
        val seqLen = seqLens[batchIdx]
        require(seqLen in 0..maxSeqLen) { "Sequence length must be in range $[0, $maxSeqLen], current seq length=$seqLen" }

        for (seqIdx in 0 until seqLen) {
            val inputOffset = batchIdx * maxSeqLen * blockSize + seqIdx * blockSize
            val outputOffset = batchIdx * maxSeqLen * blockSize + (seqLen - seqIdx - 1) * blockSize
            output.copyFrom(offset = outputOffset, this, startInOther = inputOffset, endInOther = inputOffset + blockSize)
        }

        for (seqIdx in seqLen until maxSeqLen) {
            val offset = batchIdx * maxSeqLen * blockSize + seqIdx * blockSize
            output.copyFrom(offset = offset, this, startInOther = offset, endInOther = offset + blockSize)
        }
    }
    return output as T
}

private fun <T : NDArrayCore> T.reverseSequenceTimeMajor(seqLens: IntArray, batchSize: Int, maxSeqLen: Int, blockSize: Int): T {
    val output = allocateNDArray(type, shape)

    for (batchIdx in 0 until batchSize) {
        val seqLen = seqLens[batchIdx]
        require(seqLen in 0..maxSeqLen) { "Sequence length must be in range $[0, $maxSeqLen], current seq length=$seqLen" }

        for (seqIdx in 0 until seqLen) {
            val inputOffset = seqIdx * batchSize * blockSize + batchIdx * blockSize
            val outputOffset = (seqLen - seqIdx - 1) * batchSize * blockSize + batchIdx * blockSize
            output.copyFrom(offset = outputOffset, this, startInOther = inputOffset, endInOther = inputOffset + blockSize)
        }

        for (seqIdx in seqLen until maxSeqLen) {
            val offset = seqIdx * batchSize * blockSize + batchIdx * blockSize
            output.copyFrom(offset = offset, this, startInOther = offset, endInOther = offset + blockSize)
        }
    }
    return output as T
}
