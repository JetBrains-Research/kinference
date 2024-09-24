package io.kinference.ndarray.extensions.broadcasting

import io.kinference.ndarray.arrays.NDArrayCore
import io.kinference.ndarray.extensions.utils.calculateBlock

internal fun interface ScalarBroadcastFun {
    operator fun invoke(leftOffset: Int, rightOffset: Int, destOffset: Int, axisToBroadcastIdx: Int)
}

internal data class BroadcastingInfo(
    val broadcastingShapes: Array<IntArray>,
    val broadcastingDestShape: IntArray,
    val destShape: IntArray,
    val broadcastingAxes: List<Int>,
    val broadcastAlongLastAxis: Boolean
) {
    companion object {
        fun create(inputs: List<NDArrayCore>): BroadcastingInfo {
            val maxSize = inputs.maxOf { it.shape.size }

            val isRequiredExpand = inputs.any { it.shape.size != maxSize }

            val expandedShapes = if (isRequiredExpand)
                Array(inputs.size) {
                    val shape = inputs[it].shape
                    val offset = maxSize - shape.size

                    IntArray(maxSize).apply {
                        fill(1, fromIndex = 0, toIndex = offset)
                        shape.copyInto(this, destinationOffset = offset)
                    }
                }
            else
                Array(inputs.size) { inputs[it].shape }

            val broadcastingAxes = mutableListOf<Int>()
            val destShape = IntArray(maxSize)

            for (axis in 0 until maxSize) {
                val dim = expandedShapes.first()[axis]
                destShape[axis] = dim
                for (shapeIdx in 1 until expandedShapes.size) {
                    if (dim != expandedShapes[shapeIdx][axis]) {
                        broadcastingAxes.add(axis)
                        destShape[axis] = maxOf(dim, expandedShapes[shapeIdx][axis])
                        break
                    }
                }
            }

            val broadcastAlongLastAxis = broadcastingAxes.isNotEmpty() && broadcastingAxes.last() == maxSize - 1

            val broadcastingInputShapes = Array(expandedShapes.size) {
                shapeToBroadcastingShape(expandedShapes[it], broadcastingAxes, broadcastAlongLastAxis)
            }

            val broadcastingDestShape = shapeToBroadcastingShape(destShape, broadcastingAxes, broadcastAlongLastAxis)

            return BroadcastingInfo(
                broadcastingInputShapes,
                broadcastingDestShape,
                destShape,
                broadcastingAxes,
                broadcastAlongLastAxis
            )
        }

        private fun shapeToBroadcastingShape(expandedShape: IntArray, broadcastingAxes: List<Int>, broadcastAlongLastAxis: Boolean): IntArray {
            val newShape = if (broadcastAlongLastAxis)
                IntArray(2 * broadcastingAxes.size)
            else
                IntArray(2 * broadcastingAxes.size + 1)

            var prevAxis = 0
            for (broadcastingAxisIdx in broadcastingAxes.indices) {
                val broadcastingAxis = broadcastingAxes[broadcastingAxisIdx]

                val batch = expandedShape.calculateBlock(fromIdx = prevAxis, toIdx = broadcastingAxis)
                val dim = expandedShape[broadcastingAxis]
                newShape[broadcastingAxisIdx * 2] = batch
                newShape[broadcastingAxisIdx * 2 + 1] = dim

                prevAxis = broadcastingAxis + 1
            }

            if (!broadcastAlongLastAxis) {
                val row = expandedShape.calculateBlock(fromIdx = prevAxis)
                newShape[newShape.lastIndex] = row
            }

            return newShape
        }
    }
}

internal fun makeOffsets(shape: IntArray, blocksInRow: Int): IntArray {
    val offsets = IntArray(shape.size)
    offsets[offsets.lastIndex - 1] = blocksInRow
    offsets[offsets.lastIndex] = 1

    for (idx in offsets.lastIndex - 2 downTo 0) {
        offsets[idx] = offsets[idx + 1] * shape[idx + 1]
    }

    return offsets
}
