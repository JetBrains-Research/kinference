@file:GeneratePrimitives(DataType.ALL)
package io.kinference.ndarray.extensions.reduce.primitive

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.PrimitiveNDArray
import io.kinference.ndarray.arrays.pointers.accept
import io.kinference.ndarray.arrays.pointers.forEach
import io.kinference.primitives.annotations.GenerateNameFromPrimitives
import io.kinference.primitives.annotations.GeneratePrimitives
import io.kinference.primitives.types.*

@GenerateNameFromPrimitives
internal suspend fun PrimitiveNDArray.reduceOperationPrimitive(
    axes: IntArray,
    keepDims: Boolean,
    initOutputValue: PrimitiveType? = null,
    operation: (output: PrimitiveType, input: PrimitiveType) -> PrimitiveType
): PrimitiveNDArray {
    if (axes.isEmpty()) return this

    val axesToReduce = axes.map { indexAxis(it) }.toSet()
    require(axesToReduce.all { it in shape.indices }) { "Axes ${axes.joinToString()} must be in range [-${rank}, ${rank - 1}]" }

    val outputShapeWithKeepDims = this.shape.copyOf().apply { axesToReduce.forEach { set(it, 1) } }
    val stridesWithKeepDims = Strides(outputShapeWithKeepDims)


    val outputShape = if (keepDims) outputShapeWithKeepDims else shape.sliceArray(shape.indices.minus(axesToReduce))
    val outputArray = MutablePrimitiveNDArray(outputShape)
    if (initOutputValue != null) {
        outputArray.fill(initOutputValue)
    }

    val axisToStop = axesToReduce.maxOrNull()!! + 1
    val blockToApply = this.computeBlockSize(fromDim = axisToStop)

    fun reduceOperationRecurrent(axis: Int = 0, inputOffset: Int = 0, outputOffset: Int = 0) {
        when(axis) {
            axisToStop -> {
                val inputPointer = this.array.pointer(inputOffset)
                val outputPointer = outputArray.array.pointer(outputOffset)

                outputPointer.accept(inputPointer, blockToApply) { dst: PrimitiveType, src: PrimitiveType -> operation(dst, src) }
            }
            this.shape.lastIndex -> {
                val dim = this.shape[axis]
                val inputPointer = this.array.pointer(inputOffset)
                val outputPointer = outputArray.array.pointer(outputOffset)

                var accumulator = outputPointer.get()
                inputPointer.forEach(dim) { accumulator = operation(accumulator, it) }
                outputPointer.set(accumulator)
            }
            else -> {
                val dim = this.shape[axis]
                repeat(dim) {
                    val inputAdditionalOffset = this.strides.strides[axis] * it
                    val outputAdditionalOffset = if (axis in axesToReduce) 0 else stridesWithKeepDims.strides[axis] * it

                    reduceOperationRecurrent(axis + 1, inputOffset + inputAdditionalOffset, outputOffset + outputAdditionalOffset)
                }
            }
        }
    }

    reduceOperationRecurrent()

    return outputArray
}

