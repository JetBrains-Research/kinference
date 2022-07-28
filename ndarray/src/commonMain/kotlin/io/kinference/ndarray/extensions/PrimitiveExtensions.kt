@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.ndarray.extensions

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.*
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import kotlinx.coroutines.launch
import kotlin.coroutines.CoroutineContext
import kotlin.coroutines.EmptyCoroutineContext
import kotlin.math.*

fun erf(value: PrimitiveType): PrimitiveType {
    val sign = value.toDouble().sign
    val doubleValue = abs(value.toDouble())
    val t = 1 / (1 + ERF_P_VALUE * doubleValue)

    val sum = t * (ERF_COEF[0] + t * (ERF_COEF[1] + t * (ERF_COEF[2] + t * (ERF_COEF[3] + t * ERF_COEF[4]))))

    return (sign * (1.0 - sum * exp(-doubleValue * doubleValue))).toPrimitive()
}


@SpecifyPrimitives(include = [DataType.BYTE, DataType.UBYTE])
@BindPrimitives(type1 = [DataType.BYTE, DataType.UBYTE])
fun PrimitiveNDArray.quantizeDot(other: @BindPrimitives.Type1 PrimitiveNDArray, destination: MutableFloatNDArray, zeroPointA: Int = 0, zeroPointB: Int = 0, scale: Float = 1f, coroutineContext: CoroutineContext = EmptyCoroutineContext): MutableFloatNDArray {
    val M = this.shape[0]

    fun wrapper(body: (inner: () -> Unit) -> Unit = { it() }) {
        for (rdBlockNum in 0 until destination.blocksInRow) {
            body {
                for (i in 0 until M) {
                    val dBlockOffset = i * destination.blocksInRow
                    val lBlockOffset = i * this.blocksInRow

                    var k = 0
                    for (lBlockNum in 0 until this.blocksInRow) {
                        val lBlock = this.array.blocks[lBlockOffset + lBlockNum]
                        for (lInd in lBlock.indices) {
                            val temp = lBlock[lInd].toInt() - zeroPointA
                            val rBlockOffset = k * other.blocksInRow
                            val rBlock = other.array.blocks[rBlockOffset + rdBlockNum]
                            val dBlock = destination.array.blocks[dBlockOffset + rdBlockNum]
                            for (idx in rBlock.indices) {
                                dBlock[idx] += (temp * (rBlock[idx].toInt() - zeroPointB)) * scale
                            }
                            k++
                        }
                    }
                }
            }
        }
    }

    if (other.blocksInRow > 1) {
        runBlocking(coroutineContext) { wrapper { launch { it() } } }
    } else {
        wrapper()
    }

    return destination
}
