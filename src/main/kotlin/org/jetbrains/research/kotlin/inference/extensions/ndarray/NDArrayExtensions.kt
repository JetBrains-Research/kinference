package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.*
import org.jetbrains.research.kotlin.inference.extensions.functional.*
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.types.TensorInfo
import org.jetbrains.research.kotlin.inference.types.TensorShape
import kotlin.collections.toIntArray

fun <T> TypedNDArray<T>.shouldCopy() = this !is MutableTypedNDArray

fun <T> MutableTypedNDArray<T>.wrapOneDim(): MutableTypedNDArray<T> {
    return this.reshape(1.concat(this.shape))
}

fun <T> TypedNDArray<T>.isScalar() = shape.isEmpty()

fun <T> TypedNDArray<T>.indexAxis(axis: Int): Int {
    return if (axis < 0) rank + axis else axis
}

fun <T> TypedNDArray<T>.asTensor(name: String? = null) = Tensor(this as NDArray<Any>, TensorInfo(name ?: "", type, TensorShape(this.shape)))

fun <T> MutableTypedNDArray<T>.squeeze(vararg axes: Int): MutableTypedNDArray<T> {
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

fun <T> MutableTypedNDArray<T>.unsqueeze(vararg axes: Int): TypedNDArray<T> {
    val actualAxes = axes.map { indexAxis(it) }.sorted()
    val newShape = shape.toMutableList()
    for (axis in actualAxes) {
        newShape.add(axis, 1)
    }
    return reshape(newShape.toIntArray())
}

fun <T> MutableTypedNDArray<T>.transpose(permutations: List<Number>? = null): MutableTypedNDArray<T> {
    if (rank == 2) return this.matrixTranspose()

    require(permutations.isNullOrEmpty() || permutations.size == rank) { "Axes permutations list size should match the number of axes" }
    val actualPerm = if (permutations.isNullOrEmpty()) shape.indices.reversed() else permutations.toIntArray()

    return this.transpose(actualPerm)
}

//if axis not 0
fun <T> TypedNDArray<T>.mergeOnAxis(other: TypedNDArray<T>, axis: Int): MutableTypedNDArray<T> {
    val rows = this.rows.zip(other.rows) { fst, snd -> fst.concatenate(snd, axis - 1) }.toTypedArray()
    var result = rows[0]

    val dim = this.shape[0]
    if (dim > 1) {
        result = rows.apply { set(0, rows[0].wrapOneDim()) }.reduce { acc, tensor -> acc.concatenate(tensor.wrapOneDim()) }
    }
    if (dim == 1 && axis > 0) result = result.wrapOneDim()

    return result
}

fun <T> TypedNDArray<T>.concatenate(other: TypedNDArray<T>, axis: Int = 0): MutableTypedNDArray<T> {
    val actualAxis = this.indexAxis(axis)
    if (actualAxis != 0) return this.mergeOnAxis(other, actualAxis)

    val fstDim: IntArray = this.shape
    var sndDim: IntArray = other.shape
    if (fstDim.size > 1 && sndDim.size == 1) sndDim = intArrayOf(1, sndDim[0])

    val newShape: IntArray = if (fstDim.size == 1) {
        intArrayOf(fstDim[0] + sndDim[0])
    } else {
        fstDim.copyOf(fstDim.size).apply { set(0, fstDim[0] + sndDim[0]) }
    }
    return allocateNDArray<T>(type, Strides(newShape)).apply {
        placeAll(0, this@concatenate.array)
        placeAll(this@concatenate.linearSize, other.array)
    }
}

fun <T> Collection<TypedNDArray<T>>.concatenate(axis: Int): TypedNDArray<T> {
    return this.reduce { acc, tensor -> acc.concatenate(tensor, axis) }
}

fun Array<NDArray<Any>>.stack(axis: Int): TypedNDArray<Any> {
    val fstShape = this.first().shape
    val newShape = IntArray(fstShape.size + 1)
    fstShape.copyInto(newShape, 0, 0, axis)
    newShape[axis] = 1
    fstShape.copyInto(newShape, axis + 1, axis)
    return this.map { it.toMutable().reshape(newShape) }.concatenate(axis)
}

fun <T> TypedNDArray<T>.as2DList(): List<MutableTypedNDArray<T>> {
    if (this.rank == 2) return listOf(this.toMutable())
    if (this.rank == 1) return listOf(this.toMutable().wrapOneDim())

    val matrixShape = intArrayOf(shape[indexAxis(-2)], shape[indexAxis(-1)])
    val matrixStrides = Strides(matrixShape)
    val matrixSize = matrixStrides.linearSize

    return List(strides.linearSize / matrixSize) { index ->
        allocateNDArray<T>(type, matrixStrides).apply {
            val start = matrixSize * index
            place(0, this@as2DList.array, start, start + matrixSize)
        }
    }
}

fun <T> MutableTypedNDArray<T>.reshape(tensorShape: TypedNDArray<T>): MutableTypedNDArray<T> {
    val requestedShape = tensorShape.array as LongArray
    val newShape = IntArray(requestedShape.size) { i -> requestedShape[i].toInt() }
    require(newShape.count { it == -1 } <= 1) { "At most one dimension of the new shape can be -1" }

    for ((i, axisShape) in newShape.withIndex()) {
        if (axisShape == 0) newShape[i] = shape[i]
    }

    val negativeIdx = requestedShape.indexOf(-1)
    if (negativeIdx != -1) {
        val elementsCount = newShape.filter { it != -1 }.reduce(Int::times)
        newShape[negativeIdx] = strides.linearSize / elementsCount
    }

    return reshape(newShape)
}

@Suppress("UNCHECKED_CAST")
fun <T : Any, V : Any> TypedNDArray<T>.combineWith(other: TypedNDArray<T>, transform: PrimitiveArrayValueCombineFunction<T, V>): TypedNDArray<T> {
    if (this.isScalar()) {
        return other.scalarOp(this[0] as V, transform)
    } else if (other.isScalar()) {
        return this.scalarOp(other[0] as V, transform)
    }

    transform as PrimitiveArraysCombineFunction<T>
    if (!shape.contentEquals(other.shape)) {
        return this.applyWithBroadcast(other, transform)
    }

    return if (shouldCopy()) {
        transform.apply(array, other.array); this
    } else {
        NDArray(transform.apply(array, other.array), type, strides)
    }
}
