package io.kinference.ndarray.extensions.reverse

import io.kinference.ndarray.arrays.NDArrayCore
import io.kinference.ndarray.arrays.computeBlockSize
import io.kinference.ndarray.extensions.allocateNDArray

fun <T : NDArrayCore> T.reverseSeq(mode: ReverseSeqMode, seqLens: IntArray): T {
    val numBatches = if (mode == ReverseSeqMode.BatchMajorMode) shape[0] else shape[1]
    val maxSeqLen = if (mode == ReverseSeqMode.BatchMajorMode) shape[1] else shape[0]

    require(seqLens.size == numBatches) { "Sequence lengths array size must have $numBatches elements but the array of size ${seqLens.size} was found" }

    val blockSize = computeBlockSize(fromDim = 2)
    val output = allocateNDArray(type, shape)

    for (batchIdx in 0 until numBatches) {
        val seqLength = seqLens[batchIdx]
        require(seqLength in 0..maxSeqLen) { "Sequence length must be in range $[0, $maxSeqLen], current seq length=$seqLength" }

        for (seqIdx in 0 until seqLength) {
            val inputOffset = mode.index(batchIdx, seqIdx, numBatches, maxSeqLen, blockSize)
            val outputOffset = mode.reverseIndex(batchIdx, seqIdx, seqLength, numBatches, maxSeqLen, blockSize)
            output.copyFrom(offset = outputOffset, this, startInOther = inputOffset, endInOther = inputOffset + blockSize)
        }

        for (seqIdx in seqLength until maxSeqLen) {
            val offset = mode.index(batchIdx, seqIdx, numBatches, maxSeqLen, blockSize)
            output.copyFrom(offset = offset, this, startInOther = offset, endInOther = offset + blockSize)
        }
    }
    return output as T
}
