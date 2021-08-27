package io.kinference.ndarray.extensions

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.broadcasting.Broadcasting
import io.kinference.ndarray.broadcasting.Broadcasting.broadcastShape
import io.kinference.primitives.types.DataType
import kotlin.collections.toIntArray
import kotlin.ranges.reversed

fun NDArray.isScalar() = shape.isEmpty()

fun NDArray.canDequantizePerAxis(axis: Int, zeroPoint: NDArray?, scale: NDArray): Boolean {
    return scale.rank == 1 && scale.linearSize == shape[axis] && (zeroPoint == null || zeroPoint.rank == 1 && zeroPoint.linearSize == shape[axis])
}

fun canDequantizePerTensor(zeroPoint: NDArray?, scale: NDArray): Boolean {
    return scale.linearSize == 1 && (zeroPoint == null || zeroPoint.linearSize == 1)
}

fun MutableNDArray.wrapOneDim(): MutableNDArray {
    return this.reshape(1.concat(this.shape))
}

fun NDArray.indexAxis(axis: Int): Int {
    return if (axis < 0) rank + axis else axis
}

val NDArray.rows: Array<MutableNDArray>
    get() = Array(shape[0]) { i -> row(i) }

val NumberNDArray.rows: Array<MutableNumberNDArray>
    get() = Array(shape[0]) { i -> row(i) }

fun MutableNDArray.squeeze(vararg axes: Int): MutableNDArray {
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

fun MutableNDArray.unsqueeze(vararg axes: Int): MutableNDArray {
    val actualAxes = axes.map { indexAxis(it) }.sorted()
    val newShape = shape.toMutableList()
    for (axis in actualAxes) {
        newShape.add(axis, 1)
    }

    return reshape(newShape.toIntArray())
}

fun MutableNDArray.transpose(permutations: IntArray? = null): MutableNDArray {
    require(permutations.isNullOrEmpty() || permutations!!.size == rank) { "Axes permutations list size should match the number of axes" }
    if (this.rank == 2) return this.transpose2D()

    val actualPerm = if (permutations.isNullOrEmpty()) shape.indices.reversed().toIntArray() else permutations
    return this.transpose(actualPerm!!)
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
    return this.map { it.reshapeView(newShape) }.concatenate(axis)
}

fun NDArray.as2DList(): List<NDArray> {
    if (this.rank == 2) return listOf(this)
    if (this.rank == 1) return listOf(this.copyIfNotMutable().wrapOneDim())

    val matrixShape = intArrayOf(shape[indexAxis(-2)], shape[indexAxis(-1)])
    val matrixStrides = Strides(matrixShape)
    val matrixSize = matrixStrides.linearSize

    return List(strides.linearSize / matrixSize) { index ->
        allocateNDArray(matrixStrides).apply {
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


fun MutableNDArray.reshape(tensorShape: NDArray): MutableNDArray {
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
