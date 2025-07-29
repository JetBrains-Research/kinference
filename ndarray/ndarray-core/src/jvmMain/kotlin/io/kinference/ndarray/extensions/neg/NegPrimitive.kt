@file:GeneratePrimitives(
    DataType.BYTE,
    DataType.SHORT,
    DataType.INT,
    DataType.LONG,
    DataType.FLOAT,
    DataType.DOUBLE
)
@file:GenerateVector

package io.kinference.ndarray.extensions.neg

import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.extensions.applyElementWise
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.DataType
import io.kinference.primitives.vector.*

@GenerateNameFromPrimitives
@SpecifyPrimitives(include = [DataType.INT, DataType.LONG,DataType.FLOAT, DataType.DOUBLE])
internal suspend fun negPrimitive(array: PrimitiveNDArray): PrimitiveNDArray = array.applyElementWise { -it }

@GenerateNameFromPrimitives
@SpecifyPrimitives(include = [DataType.SHORT, DataType.BYTE])
internal suspend fun negIntegerPrimitive(array: PrimitiveNDArray): PrimitiveNDArray = array.applyElementWise { (-it).toPrimitive() }

@GenerateNameFromPrimitives
internal suspend fun vecNegPrimitive(array: PrimitiveNDArray): PrimitiveNDArray{
    val output = PrimitiveNDArray(array.strides)
    val blockSize = array.array.blockSize
    for(blockIdx in 0 until array.array.blocksNum){
        val inputBlock = array.array.blocks[blockIdx]
        val outputBlock = output.array.blocks[blockIdx]
        Neg(PrimitiveSlice(inputBlock)).into(outputBlock, 0, blockSize)
    }
    return output
}
