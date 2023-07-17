@file:GeneratePrimitives(
    DataType.BYTE,
    DataType.SHORT,
    DataType.INT,
    DataType.LONG,
    DataType.UBYTE,
    DataType.USHORT,
    DataType.UINT,
    DataType.ULONG,
)

package io.kinference.ndarray.extensions.bitwise.not

import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.stubs.inv
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.DataType
import kotlin.experimental.inv

fun PrimitiveNDArray.bitNot(): PrimitiveNDArray {
    val output = MutablePrimitiveNDArray(this.strides)

    val outputIter = output.array.blocks.iterator()
    val inputIter = this.array.blocks.iterator()
    val blocksNum = this.array.blocksNum

    repeat(blocksNum) {
        val inputBlock = inputIter.next()
        val outputBlock = outputIter.next()

        for (idx in outputBlock.indices) {
            outputBlock[idx] = inputBlock[idx].inv()
        }
    }

    return output
}
