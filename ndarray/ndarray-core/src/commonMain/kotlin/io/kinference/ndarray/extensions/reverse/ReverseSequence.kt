package io.kinference.ndarray.extensions.reverse

import io.kinference.ndarray.arrays.NDArrayCore
import io.kinference.ndarray.arrays.computeBlockSize
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.utils.PlatformUtils
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlin.math.ceil
import kotlin.math.min

suspend fun <T : NDArrayCore> T.reverseSeq(mode: ReverseSeqMode, seqLens: IntArray): T {
    val numBatches = if (mode == ReverseSeqMode.BatchMajorMode) shape[0] else shape[1]
    val maxSeqLen = if (mode == ReverseSeqMode.BatchMajorMode) shape[1] else shape[0]

    require(seqLens.size == numBatches) { "Sequence lengths array size must have $numBatches elements but the array of size ${seqLens.size} was found" }

    val blockSize = computeBlockSize(fromDim = 2)
    val output = allocateNDArray(type, shape)

    val numBatchesParallelize = ceil(numBatches.toDouble() / PlatformUtils.threads).toInt()

    coroutineScope {
        for (batchIdx in 0 until numBatches step numBatchesParallelize) {
            launch {
                for (batchIdxCoroutine in batchIdx until min(batchIdx + numBatchesParallelize, numBatches)) {
                    val seqLength = seqLens[batchIdxCoroutine]
                    require(seqLength in 0..maxSeqLen) { "Sequence length must be in range $[0, $maxSeqLen], current seq length=$seqLength" }

                    for (seqIdx in 0 until seqLength) {
                        val inputOffset = mode.index(batchIdxCoroutine, seqIdx, numBatches, maxSeqLen, blockSize)
                        val outputOffset = mode.reverseIndex(batchIdxCoroutine, seqIdx, seqLength, numBatches, maxSeqLen, blockSize)
                        output.copyFrom(offset = outputOffset, this@reverseSeq, startInOther = inputOffset, endInOther = inputOffset + blockSize)
                    }

                    for (seqIdx in seqLength until maxSeqLen) {
                        val offset = mode.index(batchIdxCoroutine, seqIdx, numBatches, maxSeqLen, blockSize)
                        output.copyFrom(offset = offset, this@reverseSeq, startInOther = offset, endInOther = offset + blockSize)
                    }
                }
            }
        }
    }
    return output as T
}
