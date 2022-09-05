package io.kinference.ndarray.extensions

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.primitives.types.DataType
import kotlin.collections.toIntArray

fun NDArray.wrapOneDim(): NDArray {
    return this.reshape(1.concat(this.shape))
}

val NDArray.rows: Array<MutableNDArray>
    get() = Array(shape[0]) { i -> row(i) }

val NumberNDArray.rows: Array<MutableNumberNDArray>
    get() = Array(shape[0]) { i -> row(i) }

fun NDArray.squeeze(vararg axes: Int): NDArray {
    val actualAxes = if (axes.isNotEmpty()) {
        axes.map { indexAxis(it) }
    } else {
        shape.withIndex().filter { it.value == 1 }.map { it.index }
    }
    require(actualAxes.all { shape[it] == 1 })

    val shapeIndices = shape.indices - actualAxes
    val newShape = shape.sliceArray(shapeIndices)

    return reshape(newShape)
}

private fun indexAxisForUnsqueeze(axis: Int, shapeSize: Int): Int {
    return if (axis < 0) shapeSize + axis else axis
}

fun NDArray.unsqueeze(vararg axes: Int): NDArray {
    val actualAxes = axes.map { indexAxisForUnsqueeze(it, this.rank + axes.size) }.sorted()
    val newShape = shape.toMutableList()
    for (axis in actualAxes) {
        newShape.add(axis, 1)
    }

    return reshape(newShape.toIntArray())
}

fun Collection<NDArray>.concatenate(axis: Int): NDArray {
    return this.first().concatenate(this.drop(1), axis)
}

fun Array<NDArray>.stack(axis: Int): NDArray {
    val fstShape = this.first().shape
    val newShape = IntArray(fstShape.size + 1)
    fstShape.copyInto(newShape, 0, 0, axis)
    newShape[axis] = 1
    fstShape.copyInto(newShape, axis + 1, axis)
    return this.map { it.reshape(newShape) }.concatenate(axis)
}

fun NDArray.as2DList(): List<NDArray> {
    if (this.rank == 2) return listOf(this)
    if (this.rank == 1) return listOf(this.wrapOneDim())

    val matrixShape = intArrayOf(shape[indexAxis(-2)], shape[indexAxis(-1)])
    val matrixStrides = Strides(matrixShape)
    val matrixSize = matrixStrides.linearSize

    return List(strides.linearSize / matrixSize) { index ->
        allocateNDArray(type, matrixStrides).apply {
            val start = matrixSize * index
            copyFrom(0, this@as2DList, start, start + matrixSize)
        }
    }
}

fun viewHelper(axes: IntArray, strides: Strides): Pair<Int, IntArray> {
    val newOffset = axes.foldIndexed(0) { index, acc, i -> acc + i * strides.strides[index] }
    val newShape = strides.shape.copyOfRange(axes.size, strides.shape.size)

    return newOffset to newShape
}


fun NDArray.reshape(tensorShape: NDArray): NDArray {
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

fun NDArray.applyWithBroadcast(
    other: NDArray,
    destination: MutableNDArray,
    ordered: Boolean = false,
    op: (NDArray, NDArray, MutableNDArray) -> Unit
): MutableNDArray {
    val newShape = broadcastShape(listOf(this.shape, other.shape))

    if (ordered) require(this.shape.contentEquals(newShape))

    val opWithNewStructure = { inputs: List<NDArray>, dest: MutableNDArray -> op(inputs[0], inputs[1], dest) }

    return Broadcasting.applyWithBroadcast(listOf(this, other), destination, opWithNewStructure)
}

fun NDArray.applyWithBroadcast(
    other: NDArray,
    destType: DataType = this.type,
    ordered: Boolean = false,
    op: (NDArray, NDArray, MutableNDArray) -> Unit
): MutableNDArray {
    val newShape = broadcastShape(listOf(this.shape, other.shape))

    if (ordered) require(this.shape.contentEquals(newShape))

    val destination = allocateNDArray(destType, Strides(newShape))
    val opWithNewStructure = { inputs: List<NDArray>, dest: MutableNDArray -> op(inputs[0], inputs[1], dest) }

    return Broadcasting.applyWithBroadcast(listOf(this, other), destination, opWithNewStructure)
}

fun getIndices(indices: NDArray, axisLimit: Int): IntNDArray {
    if (indices !is IntNDArray && indices !is LongNDArray) error("Indices type must be either Long or Int. Current type = ${indices.type}")

    fun checkIndex(index: Int, axisLimit: Int): Int = if (index >= 0) index else index + axisLimit

    return if (indices is IntNDArray) {
        indices.map (object : IntMap {
            override fun apply(value: Int): Int = checkIndex(value, axisLimit)
        }) as IntNDArray
    } else {
        indices as LongNDArray
        val pointer = indices.array.pointer()
        IntNDArray(indices.shape) { checkIndex(pointer.getAndIncrement().toInt(), axisLimit) }
    }
}

fun NDArray.isTransposeReshape(permutation: IntArray): Boolean {
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

fun NumberNDArray.tryDequantize(zeroPoint: NumberNDArray?, scale: FloatNDArray, axis: Int? = null): FloatNDArray {
    require(this.type == zeroPoint?.type) { "Input data and zero point should have the same data type." }
    return when {
        this is ByteNDArray && zeroPoint is ByteNDArray -> this.dequantize(zeroPoint, scale, axis)
        this is UByteNDArray && zeroPoint is UByteNDArray -> this.dequantize(zeroPoint, scale, axis)
        else -> error("Dequantization is only supported for BYTE and UBYTE types. Current type = ${this.type}.")
    }
}

fun NumberNDArray.tryZeroPoint(zeroPoint: NumberNDArray): IntNDArray {
    require(this.type == zeroPoint.type) { "Input data and zero point should have the same data type." }
    return when {
        this is ByteNDArray && zeroPoint is ByteNDArray -> this.withZeroPoint(zeroPoint)
        this is UByteNDArray && zeroPoint is UByteNDArray -> this.withZeroPoint(zeroPoint)
        this is IntNDArray && zeroPoint is IntNDArray -> this.withZeroPoint(zeroPoint)
        else -> error("Zero point is only supported for BYTE, UBYTE and INT types. Current type = ${this.type}.")
    }
}
