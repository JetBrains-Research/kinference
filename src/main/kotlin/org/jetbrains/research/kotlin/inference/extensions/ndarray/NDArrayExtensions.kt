package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.applyWithBroadcast
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveCombineFunction
import org.jetbrains.research.kotlin.inference.extensions.primitives.*

fun <T> NDArray<T>.wrapOneDim(): NDArray<T> {
    val newStrides = Strides(intArrayOf(1, *this.shape))
    return this.clone(newStrides)
}

//if axis not 0
fun <T> NDArray<T>.mergeOnAxis(other: NDArray<T>, axis: Int): NDArray<T> {
    val rows = this.rows.zip(other.rows) { fst, snd -> fst.concatenate(snd, axis - 1) }.toTypedArray()
    var result = rows[0]

    val dim = this.shape[0]
    if (dim > 1) {
        result = rows.apply { set(0, rows[0].wrapOneDim()) }.reduce { acc, tensor -> acc.concatenate(tensor.wrapOneDim()) }
    }
    if (dim == 1 && axis > 0) result = result.wrapOneDim()

    return result
}

fun <T> NDArray<T>.concatenate(other: NDArray<T>, axis: Int = 0): NDArray<T> {
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
    return allocateNDArray(type, Strides(newShape)).apply {
        placeAll(0, this@concatenate.array)
        placeAll(this@concatenate.linearSize, other.array)
    } as NDArray<T>
}

fun <T> Collection<NDArray<T>>.concatenate(axis: Int): NDArray<T> {
    return this.reduce { acc, tensor -> acc.concatenate(tensor, axis) }
}

fun Array<NDArray<Any>>.stack(axis: Int): NDArray<Any> {
    val fstShape = this.first().shape
    val newShape = IntArray(fstShape.size + 1)
    fstShape.copyInto(newShape, 0, 0, axis)
    newShape[axis] = 1
    fstShape.copyInto(newShape, axis + 1, axis)
    return this.map { it.reshape(newShape) }.concatenate(axis)
}

fun <T> NDArray<T>.as2DList(): List<NDArray<T>> {
    if (this.rank == 2) return listOf(this)
    if (this.rank == 1) return listOf(this.wrapOneDim())

    val matrixShape = intArrayOf(shape[indexAxis(-2)], shape[indexAxis(-1)])
    val matrixStrides = Strides(matrixShape)
    val matrixSize = matrixStrides.linearSize

    return List(strides.linearSize / matrixSize) { index ->
        allocateNDArray(type, matrixStrides).apply {
            placeAll(0, this@as2DList.slice(matrixSize, matrixSize * index))
        } as NDArray<T>
    }
}

fun <T> NDArray<T>.reshape(tensorShape: NDArray<T>): NDArray<T> {
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

inline fun <reified T : Any> NDArray<T>.combineWith(other: NDArray<T>, transform: PrimitiveCombineFunction<T>): NDArray<T> {
    if (this.isScalar()) {
        return other.scalarOp(this[0], transform)
    } else if (other.isScalar()) {
        return this.scalarOp(other[0], transform)
    }

    if (!shape.contentEquals(other.shape)) {
        return applyWithBroadcast(other, transform)
    }

    return NDArray(transform.apply(array, other.array), type, strides)
}
