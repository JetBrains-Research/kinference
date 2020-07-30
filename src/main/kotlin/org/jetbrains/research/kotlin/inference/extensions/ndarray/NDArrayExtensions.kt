package org.jetbrains.research.kotlin.inference.extensions.ndarray

import org.jetbrains.research.kotlin.inference.data.ndarray.LongNDArray
import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.applyWithBroadcast
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import kotlin.math.min

inline fun <reified T> NDArray<T>.splitWithAxis(parts: Int, axis: Int = 0, keepDims: Boolean = true): List<NDArray<T>> {
    require(axis in shape.indices) { "Index $axis out of shape bound: (0, ${rank - 1}" }

    val elementsByIndex = shape[axis]
    val mainSplit = elementsByIndex / parts
    val split = MutableList(parts - 1) { mainSplit }

    val tail = elementsByIndex - mainSplit * (parts - 1)
    split.add(tail)

    return this.splitWithAxis(split.toIntArray(), axis, keepDims).toList()
}

inline fun <reified T> NDArray<T>.splitWithAxis(splitTensor: NDArray<T>, axis: Int = 0, keepDims: Boolean = true): List<NDArray<T>> {
    return if (splitTensor.linearSize == 1) {
        splitWithAxis((splitTensor[0] as Number).toInt(), axis, keepDims)
    } else {
        this.splitWithAxis((splitTensor.array as List<Long>).toIntArray(), axis, keepDims).toList()
    }
}

fun <T> NDArray<T>.wrapOneDim(): NDArray<T> {
    val newStrides = Strides(intArrayOf(1, *this.shape))
    return this.clone(newStrides)
}

//if axis not 0
fun <T> NDArray<T>.mergeOnAxis(other: NDArray<T>, axis: Int): NDArray<T> {
    val dim = this.shape
    val rows = this.rows.zip(other.rows).map { (fst, snd) -> fst.concatenate(snd, axis - 1) }.toMutableList()
    var result = rows[0]

    if (dim[0] > 1) {
        result = rows.apply { set(0, rows[0].wrapOneDim()) }.reduce { acc, tensor -> acc.concatenate(tensor.wrapOneDim()) }
    }
    if (dim[0] == 1 && axis > 0) result = result.wrapOneDim()

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
    return this.map {
        val newShape = this.first().shape.toMutableList()
        newShape.add(axis, 1)
        it.reshape(newShape.toIntArray())
    }.concatenate(axis)
}

fun <T> NDArray<T>.as2DList(): List<NDArray<T>> {
    if (this.rank == 2) return listOf(this)
    if (this.rank == 1) return listOf(this.wrapOneDim())

    val matrixShape = intArrayOf(shape[indexAxis(-2)], shape[indexAxis(-1)])
    val matrixStrides = Strides(matrixShape)

    return List(strides.linearSize / matrixStrides.linearSize) { index ->
        createNDArray<T>(type, matrixStrides) {
            this[it + index * matrixStrides.linearSize]
        }
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

inline fun <reified T : Any> NDArray<T>.combineWith(other: NDArray<T>, noinline transform: (T, T) -> T): NDArray<T> {
    if (this.isScalar()) {
        return other.scalarOp(this[0], transform)
    } else if (other.isScalar()) {
        return this.scalarOp(other[0], transform)
    }

    if (!shape.contentEquals(other.shape)) {
        return applyWithBroadcast(other, transform)
    }

    return NDArray(transform(array, other.array), type, strides)
}
