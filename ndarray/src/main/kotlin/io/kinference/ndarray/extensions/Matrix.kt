package io.kinference.ndarray.extensions

import io.kinference.ndarray.*

fun gemm(m: Int, n: Int, k: Int, alpha: Double, a: NumberNDArray, b: NumberNDArray, beta: Double, c: MutableNDArray,
         aOffset: Int = 0, bOffset: Int = 0, cOffset: Int = 0, transposeA: Boolean = false, transposeB: Boolean = false) : MutableNDArray {
    val lda = if (transposeA) m else k
    val ldb = if (transposeB) k else n
    return a.gemm(m, n, k, alpha, lda, b, ldb, beta, c, n, aOffset, bOffset, cOffset, transposeA, transposeB)
}

private fun NumberNDArray.getOutputStrides(other: NumberNDArray): Strides {
    val outputMatrixShape = intArrayOf(shape[indexAxis(-2)], other.shape[other.indexAxis(-1)])
    val broadcastShape = broadcastShape(shape.copyOfRange(0, rank - 2), other.shape.copyOfRange(0, other.rank - 2))

    val outputShape = IntArray(broadcastShape.size + 2)
    broadcastShape.copyInto(outputShape)
    outputMatrixShape.copyInto(outputShape, broadcastShape.size)

    return Strides(outputShape)
}

infix fun NumberNDArray.matmul(other: NumberNDArray): MutableNumberNDArray {
    val outputStrides = getOutputStrides(other)
    val outputArray = allocateNDArray(outputStrides)
    return matmul(other, outputArray) { otherArray, dest -> this.dot(otherArray, dest) }
}

private fun NumberNDArray.matmul(other: NumberNDArray, dest: MutableNumberNDArray,
                         dotFunc: NumberNDArray.(NumberNDArray, MutableNumberNDArray) -> MutableNumberNDArray): MutableNumberNDArray {
    require(!this.isScalar() && !other.isScalar()) { "Matmul operation is not available for scalar tensors" }
    fun matmul(leftInfo: BroadcastingInfo,
               rightInfo: BroadcastingInfo,
               destinationInfo: MutableBroadcastingInfo,
               temp: BroadcastingTemp, index: Int) {
        if (leftInfo.array.shape.size - index == 2) {
            temp.leftTemp.copyFrom(0, leftInfo.array, leftInfo.offset, leftInfo.offset + temp.leftTemp.linearSize)
            temp.rightTemp.copyFrom(0, rightInfo.array, rightInfo.offset, rightInfo.offset + temp.rightTemp.linearSize)

            (temp.leftTemp as NumberNDArray).dotFunc(temp.rightTemp as NumberNDArray, temp.destinationTemp as MutableNumberNDArray)

            destinationInfo.array.copyFrom(destinationInfo.offset, temp.destinationTemp)
            temp.destinationTemp.clean()
        } else {
            innerBroadcast(leftInfo, rightInfo, destinationInfo, index) { fstArray, sndArray, dest -> matmul(fstArray, sndArray, dest, temp, index + 1) }
        }
    }

    if (rank <= 2 && other.rank <= 2) {
        val actualThis = if (rank == 1) this.reshapeView(1.concat(shape)) as NumberNDArray else this
        val actualOther = if (other.rank == 1) this.reshapeView(other.shape.concat(1)) else other

        return actualThis.dotFunc(actualOther as NumberNDArray, dest)
    }

    val leftWrapShape = unsqueezeFirst(shape, dest.rank)
    val rightWrapShape = unsqueezeFirst(other.shape, dest.rank)

    val leftTempMatrixShape = intArrayOf(leftWrapShape[leftWrapShape.lastIndex - 1], leftWrapShape[leftWrapShape.lastIndex])
    val rightTempMatrixShape = intArrayOf(rightWrapShape[rightWrapShape.lastIndex - 1], rightWrapShape[rightWrapShape.lastIndex])
    val destinationTempMatrixShape = intArrayOf(leftTempMatrixShape[0], rightTempMatrixShape[1])

    val leftTempMatrix = allocateNDArray(this.type, leftTempMatrixShape)
    val rightTempMatrix = allocateNDArray(other.type, rightTempMatrixShape)
    val destinationTempMatrix = allocateNDArray(dest.type, destinationTempMatrixShape)

    val broadcastingTemp = BroadcastingTemp(leftTempMatrix, rightTempMatrix, destinationTempMatrix)

    val leftWrapped = this.reshapeView(leftWrapShape)
    val rightWrapped = other.reshapeView(rightWrapShape)

    matmul(BroadcastingInfo(leftWrapped),
        BroadcastingInfo(rightWrapped),
        MutableBroadcastingInfo(dest), broadcastingTemp, 0)
    return dest
}
