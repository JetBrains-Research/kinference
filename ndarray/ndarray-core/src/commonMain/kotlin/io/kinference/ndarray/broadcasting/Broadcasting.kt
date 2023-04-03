package io.kinference.ndarray.broadcasting

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.primitives.types.DataType

// TODO remove to different module
fun unsqueezeFirst(shape: IntArray, newShapeSize: Int): IntArray {
    val wrapSize = newShapeSize - shape.size

    val wrappedShape = IntArray(newShapeSize)
    wrappedShape.fill(1, 0, wrapSize)
    shape.copyInto(wrappedShape, wrapSize)
    return wrappedShape
}

object Broadcasting {
    fun broadcastShapeForMatmul(leftShape: IntArray, rightShape: IntArray): IntArray {
        val actualLeftShape = if (leftShape.size == 1) intArrayOf(1, leftShape[0]) else leftShape
        val actualRightShape = if (rightShape.size == 1) intArrayOf(1, rightShape[1]) else rightShape

        val outputMatrixShape = intArrayOf(actualLeftShape[actualLeftShape.lastIndex - 1], actualRightShape.last())
        val broadcastShape = broadcastShape(listOf(actualLeftShape.copyOfRange(0, actualLeftShape.size - 2),
                                                   actualRightShape.copyOfRange(0, actualRightShape.size - 2)))

        val outputShape = IntArray(broadcastShape.size + 2)
        broadcastShape.copyInto(outputShape)
        outputMatrixShape.copyInto(outputShape, broadcastShape.size)

        return outputShape
    }

    suspend fun applyWithBroadcast(
        inputs: List<NDArrayCore>,
        destination: MutableNDArrayCore,
        op: suspend (List<NDArrayCore>, MutableNDArrayCore) -> Unit
    ): MutableNDArrayCore {
        val wrappedInputs = inputs.map { it.reshape(unsqueezeFirst(it.shape, destination.shape.size)) }

        broadcast(wrappedInputs, destination, op)
        return destination
    }

    suspend fun applyWithBroadcast(
        inputs: List<NDArrayCore>,
        destType: DataType,
        op: suspend (List<NDArrayCore>, MutableNDArrayCore) -> Unit
    ): MutableNDArrayCore {
        val newShape = broadcastShape(inputs.map { it.shape })
        val destination = allocateNDArray(destType, newShape)

        val wrappedInputs = inputs.map { it.reshape(unsqueezeFirst(it.shape, newShape.size)) }

        broadcast(wrappedInputs, destination, op)
        return destination
    }

    suspend fun matmulWithBroadcast(
        left: NDArrayCore,
        right: NDArrayCore,
        destination: MutableNDArrayCore,
        dotFunc: suspend NumberNDArrayCore.(NumberNDArrayCore, MutableNumberNDArrayCore) -> MutableNumberNDArrayCore
    ) {
        require(broadcastShapeForMatmul(left.shape, right.shape).contentEquals(destination.shape))

        val wrappedLeft = left.reshape(unsqueezeFirst(left.shape, destination.shape.size))
        val wrappedRight = right.reshape(unsqueezeFirst(right.shape, destination.shape.size))

        matmulBroadcast(wrappedLeft, wrappedRight, destination, dotFunc)
    }

    private suspend fun broadcast(
        inputs: List<NDArrayCore>,
        destination: MutableNDArrayCore,
        op: suspend (List<NDArrayCore>, MutableNDArrayCore) -> Unit
    ) {
        if (inputs.all { it.shape.contentEquals(destination.shape) }) { // check all shapes (inputs and destination) equals
            op(inputs, destination)
        } else {
            innerBroadcast(inputs, destination) { inputs, dest -> broadcast(inputs, dest, op) }
        }
    }

    private suspend fun matmulBroadcast(
        left: NDArrayCore,
        right: NDArrayCore,
        destination: MutableNDArrayCore,
        dotFunc: suspend NumberNDArrayCore.(NumberNDArrayCore, MutableNumberNDArrayCore) -> MutableNumberNDArrayCore
    ) {
        if (left.rank == 2) {

            (left as NumberNDArrayCore).dotFunc(right as NumberNDArrayCore, destination as MutableNumberNDArrayCore)

        } else {
            innerBroadcast(listOf(left, right), destination) { inputs, dest -> matmulBroadcast(inputs[0], inputs[1], dest, dotFunc) }
        }
    }

    private suspend fun innerBroadcast(
        inputs: List<NDArrayCore>,
        destination: MutableNDArrayCore,
        recurrentBack: suspend (List<NDArrayCore>, MutableNDArrayCore) -> Unit
    ) {
        val indexedInputs = inputs.withIndex()
        val (arraysWithOne, arraysWithoutOne) = indexedInputs.partition { it.value.shape[0] == 1 }

        if (destination.shape.size == 1) {
            val broadcastSize = destination.shape.last()
            val broadcastArraysWithOne = arraysWithOne.map {
                val value = allocateNDArray(it.value.type, Strides(intArrayOf(broadcastSize)))
                it.copy(value = value.apply { fill(it.value.singleValue()) })
            }
            val mergedInputs = broadcastArraysWithOne.plus(arraysWithoutOne).sortedBy { it.index }.map { it.value }

            return recurrentBack(mergedInputs, destination)
        }

        val viewedArraysWithOne = arraysWithOne.map { it.copy(value = it.value.view(0)) }

        for (i in 0 until destination.shape[0]) {
            val viewedArraysWithoutOne = arraysWithoutOne.map { it.copy(value = it.value.view(i)) }
            val viewedDestination = destination.viewMutable(i)

            val mergedViewedInputs = viewedArraysWithOne.plus(viewedArraysWithoutOne).sortedBy { it.index }.map { it.value }

            recurrentBack(mergedViewedInputs, viewedDestination)
        }
    }
}
