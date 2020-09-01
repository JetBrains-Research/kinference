package io.kinference.ndarray.extensions

import io.kinference.ndarray.*

fun gemm(m: Int, n: Int, k: Int, alpha: Double, a: NumberNDArray, b: NumberNDArray, beta: Double, c: MutableNDArray,
         aOffset: Int = 0, bOffset: Int = 0, cOffset: Int = 0, transposeA: Boolean = false, transposeB: Boolean = false) : MutableNDArray {
    val lda = if (transposeA) m else k
    val ldb = if (transposeB) k else n
    return a.gemm(m, n, k, alpha, lda, b, ldb, beta, c, n, aOffset, bOffset, cOffset, transposeA, transposeB)
}


infix fun NumberNDArray.matmul(other: NumberNDArray): MutableNumberNDArray {
    require(!this.isScalar() && !other.isScalar()) { "Matmul operation is not available for scalar tensors" }
    fun matmul(left: NDArray, right: NDArray, destination: MutableNDArray) {
        if (left.shape.size == 2) {
            (left as NumberNDArray).dot(right as NumberNDArray, destination as MutableNumberNDArray)
        } else {
            innerBroadcast(left, right, destination, ::matmul)
        }
    }

    if (rank <= 2 && other.rank <= 2) {
        val actualThis: NumberNDArray = if (rank == 1) this.toMutable().reshape(1.concat(shape)) else this
        val actualOther = if (other.rank == 1) this.toMutable().reshape(other.shape.concat(1)) else other

        val newStrides = Strides(intArrayOf(shape[0], other.shape[1]))
        val destination = allocateNDArray(newStrides)

        return actualThis.dot(actualOther, destination)
    }

    val outputMatrixShape = intArrayOf(shape[indexAxis(-2)], other.shape[other.indexAxis(-1)])
    val broadcastShape = broadcastShape(shape.copyOfRange(0, rank - 2), other.shape.copyOfRange(0, other.rank - 2))

    val outputShape = IntArray(broadcastShape.size + 2)
    broadcastShape.copyInto(outputShape)
    outputMatrixShape.copyInto(outputShape, broadcastShape.size)

    val outputStrides = Strides(outputShape)
    val outputArray = allocateNDArray(outputStrides)

    val leftWrapShape = unsqueezeFirst(shape, outputShape.size)
    val rightWrapShape = unsqueezeFirst(other.shape, outputShape.size)

    val leftWrapped = this.toMutable(Strides(leftWrapShape))
    val rightWrapped = other.toMutable(Strides(rightWrapShape))

    matmul(leftWrapped, rightWrapped, outputArray)
    return outputArray
}
