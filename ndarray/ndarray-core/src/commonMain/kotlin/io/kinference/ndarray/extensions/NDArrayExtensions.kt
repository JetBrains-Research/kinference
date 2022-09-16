package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.ndarray.concat
import io.kinference.primitives.types.DataType

fun NDArrayCore.wrapOneDim(): NDArray {
    return this.reshape(1.concat(this.shape))
}

fun squeeze(array: NDArrayCore, vararg axes: Int): NDArrayCore {
    val actualAxes = if (axes.isNotEmpty()) {
        axes.map { array.indexAxis(it) }
    } else {
        array.shape.withIndex().filter { it.value == 1 }.map { it.index }
    }
    require(actualAxes.all { array.shape[it] == 1 })

    val shapeIndices = array.shape.indices - actualAxes
    val newShape = array.shape.sliceArray(shapeIndices)

    return array.reshape(newShape)
}

fun unsqueeze(array: NDArrayCore, vararg axes: Int): NDArrayCore {
    fun indexAxisForUnsqueeze(axis: Int, shapeSize: Int): Int = if (axis < 0) shapeSize + axis else axis

    val actualAxes = axes.map { indexAxisForUnsqueeze(it, array.rank + axes.size) }.sorted()
    val newShape = array.shape.toMutableList()
    for (axis in actualAxes) {
        newShape.add(axis, 1)
    }

    return array.reshape(newShape.toIntArray())
}

fun NDArrayCore.reshape(tensorShape: NDArray): NDArrayCore {
    require(tensorShape is LongNDArray) { "Tensor shape must have Long type" }

    val pointer = tensorShape.array.pointer()
    val newShape = IntArray(tensorShape.linearSize) { pointer.getAndIncrement().toInt() }
    require(newShape.count { it == -1 } <= 1) { "At most one dimension of the new shape can be -1" }

    for ((i, axisShape) in newShape.withIndex()) {
        if (axisShape == 0) newShape[i] = shape[i]
    }

    val negativeIdx = newShape.indexOf(-1)
    if (negativeIdx != -1) {
        val elementsCount = newShape.filter { it != -1 }.fold(1, Int::times)
        newShape[negativeIdx] = strides.linearSize / elementsCount
    }

    return reshape(newShape)
}

fun viewHelper(axes: IntArray, strides: Strides): Pair<Int, IntArray> {
    val newOffset = axes.foldIndexed(0) { index, acc, i -> acc + i * strides.strides[index] }
    val newShape = strides.shape.copyOfRange(axes.size, strides.shape.size)

    return newOffset to newShape
}

fun NDArrayCore.applyWithBroadcast(
    other: NDArrayCore,
    destination: MutableNDArrayCore,
    ordered: Boolean = false,
    op: (NDArrayCore, NDArrayCore, MutableNDArrayCore) -> Unit
): MutableNDArray {
    val newShape = broadcastShape(listOf(this.shape, other.shape))

    if (ordered) require(this.shape.contentEquals(newShape))

    val opWithNewStructure = { inputs: List<NDArrayCore>, dest: MutableNDArrayCore -> op(inputs[0], inputs[1], dest) }

    return Broadcasting.applyWithBroadcast(listOf(this, other), destination, opWithNewStructure)
}

fun NDArrayCore.applyWithBroadcast(
    other: NDArrayCore,
    destType: DataType = this.type,
    ordered: Boolean = false,
    op: (NDArrayCore, NDArrayCore, MutableNDArrayCore) -> Unit
): MutableNDArrayCore {
    val newShape = broadcastShape(listOf(this.shape, other.shape))

    if (ordered) require(this.shape.contentEquals(newShape))

    val destination = allocateNDArray(destType, Strides(newShape))
    val opWithNewStructure = { inputs: List<NDArrayCore>, dest: MutableNDArrayCore -> op(inputs[0], inputs[1], dest) }

    return Broadcasting.applyWithBroadcast(listOf(this, other), destination, opWithNewStructure)
}

fun NDArrayCore.isTransposeReshape(permutation: IntArray): Boolean {
    var lastPermutedAxis = 0
    for (idx in permutation.indices) {
        if (shape[permutation[idx]] == 1) {
            continue
        }
        if (permutation[idx] < lastPermutedAxis)
            return false
        lastPermutedAxis = permutation[idx]
    }
    return true
}

fun NumberNDArrayCore.tryDequantize(zeroPoint: NumberNDArrayCore?, scale: FloatNDArray, axis: Int? = null): FloatNDArray {
    require(this.type == zeroPoint?.type) { "Input data and zero point should have the same data type." }
    return when {
        this is ByteNDArray && zeroPoint is ByteNDArray -> this.dequantize(zeroPoint, scale, axis)
        this is UByteNDArray && zeroPoint is UByteNDArray -> this.dequantize(zeroPoint, scale, axis)
        else -> error("Dequantization is only supported for BYTE and UBYTE types. Current type = $type.")
    }
}

fun NumberNDArrayCore.tryZeroPoint(zeroPoint: NumberNDArrayCore): IntNDArray {
    require(this.type == zeroPoint.type) { "Input data and zero point should have the same data type." }
    return when {
        this is ByteNDArray && zeroPoint is ByteNDArray -> this.withZeroPoint(zeroPoint)
        this is UByteNDArray && zeroPoint is UByteNDArray -> this.withZeroPoint(zeroPoint)
        this is IntNDArray && zeroPoint is IntNDArray -> this.withZeroPoint(zeroPoint)
        else -> error("Zero point is only supported for BYTE, UBYTE and INT types. Current type = $type.")
    }
}
