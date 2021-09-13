package io.kinference.ndarray.extensions

import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.ndarray.concat
import io.kinference.primitives.types.DataType

fun gemm(m: Int, n: Int, k: Int, alpha: Double, a: NumberNDArray, b: NumberNDArray, beta: Double, c: MutableNDArray,
         aOffset: Int = 0, bOffset: Int = 0, cOffset: Int = 0, transposeA: Boolean = false, transposeB: Boolean = false) : MutableNDArray {
    val lda = if (transposeA) m else k
    val ldb = if (transposeB) k else n
    return a.gemm(m, n, k, alpha, lda, b, ldb, beta, c, n, aOffset, bOffset, cOffset, transposeA, transposeB)
}

infix fun NumberNDArray.matmul(other: NumberNDArray): MutableNumberNDArray {
    val outputShape = Broadcasting.broadcastShapeForMatmul(this.shape, other.shape)
    val outputArray = allocateNDArray(Strides(outputShape))
    return matmul(other, outputArray) { otherArray, dest -> this.dot(otherArray, dest) }
}

private fun NumberNDArray.matmul(other: NumberNDArray, dest: MutableNumberNDArray,
                                 dotFunc: NumberNDArray.(NumberNDArray, MutableNumberNDArray) -> MutableNumberNDArray
): MutableNumberNDArray {
    require(!this.isScalar() && !other.isScalar()) { "Matmul operation is not available for scalar tensors" }

    if (rank <= 2 && other.rank <= 2) {
        val actualThis = if (rank == 1) this.reshapeView(1.concat(shape)) as NumberNDArray else this
        val actualOther = if (other.rank == 1) this.reshapeView(other.shape.concat(1)) else other

        return actualThis.dotFunc(actualOther as NumberNDArray, dest)
    }

    Broadcasting.matmulWithBroadcast(this, other, dest, dotFunc)
    return dest
}

fun quantizeMatMul(left: NumberNDArray, right: NumberNDArray, leftZeroPoint: NumberNDArray?, rightZeroPoint: NumberNDArray?,
                   leftScale: FloatNDArray, rightScale: FloatNDArray): MutableFloatNDArray {
    val outputShape = Broadcasting.broadcastShapeForMatmul(left.shape, right.shape)
    val outputArray = allocateNDArray(DataType.FLOAT, outputShape) as MutableFloatNDArray
    return quantizeMatMul(left, right, leftZeroPoint, rightZeroPoint, leftScale, rightScale, outputArray)
}

fun quantizeMatMul(left: NumberNDArray, right: NumberNDArray, leftZeroPoint: NumberNDArray?, rightZeroPoint: NumberNDArray?,
                   leftScale: FloatNDArray, rightScale: FloatNDArray, dest: MutableFloatNDArray): MutableFloatNDArray {

    if (canDequantizePerTensor(leftZeroPoint, leftScale) && canDequantizePerTensor(rightZeroPoint, rightScale)) {
        val leftScaleValue = leftScale.singleValue()
        val rightScaleValue = rightScale.singleValue()

        val fullScale = leftScaleValue * rightScaleValue

        when {
            left.type == DataType.BYTE && right.type == DataType.BYTE -> {
                val leftZeroPointValue = (leftZeroPoint?.singleValue() as? Byte)?.toInt() ?: 0
                val rightZeroPointValue = (rightZeroPoint?.singleValue() as? Byte)?.toInt() ?: 0
                left.matmul(right, dest) { other, dest ->
                    (this as ByteNDArray).quantizeDot(other as ByteNDArray, dest as MutableFloatNDArray, leftZeroPointValue, rightZeroPointValue, fullScale)
                }
            }

            left.type == DataType.BYTE && right.type == DataType.UBYTE -> {
                val leftZeroPointValue = (leftZeroPoint?.singleValue() as? Byte)?.toInt() ?: 0
                val rightZeroPointValue = (rightZeroPoint?.singleValue() as? UByte)?.toInt() ?: 0
                left.matmul(right, dest) { other, dest ->
                    (this as ByteNDArray).quantizeDot(other as UByteNDArray, dest as MutableFloatNDArray, leftZeroPointValue, rightZeroPointValue, fullScale)

                }
            }

            left.type == DataType.UBYTE && right.type == DataType.BYTE -> {
                val leftZeroPointValue = (leftZeroPoint?.singleValue() as? UByte)?.toInt() ?: 0
                val rightZeroPointValue = (rightZeroPoint?.singleValue() as? Byte)?.toInt() ?: 0
                left.matmul(right, dest) { other, dest ->
                    (this as UByteNDArray).quantizeDot(other as ByteNDArray, dest as MutableFloatNDArray, leftZeroPointValue, rightZeroPointValue, fullScale)
                }
            }

            left.type == DataType.UBYTE && right.type == DataType.UBYTE -> {
                val leftZeroPointValue = (leftZeroPoint?.singleValue() as? UByte)?.toInt() ?: 0
                val rightZeroPointValue = (rightZeroPoint?.singleValue() as? UByte)?.toInt() ?: 0
                left.matmul(right, dest) { other, dest ->
                    (this as UByteNDArray).quantizeDot(other as UByteNDArray, dest as MutableFloatNDArray, leftZeroPointValue, rightZeroPointValue, fullScale)
                }
            }
        }
    } else {
        val leftDequantized = left.dequantize(leftZeroPoint, leftScale) as MutableFloatNDArray
        val rightDequantized = right.dequantize(rightZeroPoint, rightScale) as MutableFloatNDArray

        leftDequantized.matmul(rightDequantized, dest) { other, dest -> this.dot(other, dest) }
    }

    return dest
}
