package org.jetbrains.research.kotlin.inference.math.extensions

import org.jetbrains.research.kotlin.inference.annotations.DataType
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.math.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.types.TensorInfo
import org.jetbrains.research.kotlin.inference.types.TensorShape
import kotlin.collections.toIntArray

fun TensorProto.DataType.resolveLocalDataType(): DataType {
    return when(this) {
        TensorProto.DataType.DOUBLE -> DataType.DOUBLE
        TensorProto.DataType.FLOAT, TensorProto.DataType.FLOAT16 -> DataType.FLOAT
        TensorProto.DataType.INT32 -> DataType.INT
        TensorProto.DataType.INT64 -> DataType.LONG
        TensorProto.DataType.INT16 -> DataType.SHORT
        TensorProto.DataType.INT8-> DataType.BYTE
        TensorProto.DataType.BOOL -> DataType.BOOLEAN
        TensorProto.DataType.UINT32-> DataType.UINT
        TensorProto.DataType.UINT64 -> DataType.ULONG
        TensorProto.DataType.UINT16 -> DataType.USHORT
        TensorProto.DataType.UINT8 -> DataType.UBYTE
        TensorProto.DataType.UNDEFINED -> DataType.UNKNOWN
        else -> error("Cannot resolve data type")
    }
}

fun DataType.resolveProtoDataType(): TensorProto.DataType {
    return when(this) {
        DataType.DOUBLE -> TensorProto.DataType.DOUBLE
        DataType.FLOAT -> TensorProto.DataType.FLOAT
        DataType.INT -> TensorProto.DataType.INT32
        DataType.LONG -> TensorProto.DataType.INT64
        DataType.SHORT -> TensorProto.DataType.INT16
        DataType.BYTE -> TensorProto.DataType.INT8
        DataType.BOOLEAN -> TensorProto.DataType.BOOL
        DataType.UINT -> TensorProto.DataType.UINT32
        DataType.ULONG -> TensorProto.DataType.UINT64
        DataType.USHORT -> TensorProto.DataType.UINT16
        DataType.UBYTE -> TensorProto.DataType.UINT8
        DataType.UNKNOWN -> TensorProto.DataType.UNDEFINED
        else -> error("Cannot resolve data type")
    }
}

fun NDArray.isScalar() = shape.isEmpty()

fun MutableNDArray.wrapOneDim(): MutableNDArray {
    return this.reshape(1.concat(this.shape))
}

fun NDArray.indexAxis(axis: Int): Int {
    return if (axis < 0) rank + axis else axis
}

fun NDArray.asTensor(name: String? = null) = Tensor(this, TensorInfo(name ?: "", type.resolveProtoDataType(), TensorShape(this.shape)))

val NDArray.rows: Array<MutableNDArray>
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

fun MutableNDArray.transpose(permutations: List<Number>? = null): MutableNDArray {
    require(permutations.isNullOrEmpty() || permutations.size == rank) { "Axes permutations list size should match the number of axes" }
    if (this.rank == 2) return this.transpose2D()

    val actualPerm = if (permutations.isNullOrEmpty()) shape.indices.reversed() else permutations.toIntArray()
    return this.transpose(actualPerm)
}

//if axis not 0
fun NDArray.mergeOnAxis(other: NDArray, axis: Int): MutableNDArray {
    val rows = this.rows.zip(other.rows) { fst, snd -> fst.concatenate(snd, axis - 1) }.toTypedArray()
    var result = rows[0]

    val dim = this.shape[0]
    if (dim > 1) {
        result = rows.apply { set(0, rows[0].wrapOneDim()) }.reduce { acc, tensor -> acc.concatenate(tensor.wrapOneDim()) }
    }
    if (dim == 1 && axis > 0) result = result.wrapOneDim()

    return result
}

fun NDArray.concatenate(other: NDArray, axis: Int = 0): MutableNDArray {
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

    return allocateNDArray(Strides(newShape)).apply {
        placeAllFrom(0, this@concatenate)
        placeAllFrom(this@concatenate.linearSize, other)
    }
}

fun Collection<NDArray>.concatenate(axis: Int): NDArray {
    return this.reduce { acc, tensor -> acc.concatenate(tensor, axis) }
}

fun Array<NDArray>.stack(axis: Int): NDArray {
    val fstShape = this.first().shape
    val newShape = IntArray(fstShape.size + 1)
    fstShape.copyInto(newShape, 0, 0, axis)
    newShape[axis] = 1
    fstShape.copyInto(newShape, axis + 1, axis)
    return this.map { it.copyIfNotMutable().reshape(newShape) }.concatenate(axis)
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
            placeFrom(0, this@as2DList, start, start + matrixSize)
        }
    }
}

fun NDArray.slice(dest: LateInitArray, offset: Int, axis: Int, shape: IntArray, starts: IntArray, ends: IntArray, steps: IntArray) {
    val start = starts[axis]
    val end = ends[axis]
    val step = steps[axis]

    val range = if (step > 0) (start until end step step) else (start downTo end + 1 step -step)

    if (axis == shape.size - 1) {
        appendToLateInitArray(dest, range, offset)
    } else {
        var dim = 1
        for (ind in (axis + 1) until shape.size) dim *= shape[ind]

        for (index in range) {
            slice(dest, offset + index * dim, axis + 1, shape, starts, ends, steps)
        }
    }
}

infix fun NumberNDArray.matmul(other: NumberNDArray): MutableNumberNDArray {
    require(!this.isScalar() && !other.isScalar()) { "Matmul operation is not available for scalar tensors" }
    fun matmul(left: NDArray, right: NDArray, destination: MutableNDArray) {
        if (left.shape.size == 2) {
            (left as NumberNDArray).dot(right as NumberNDArray, destination as MutableNumberNDArray)
        } else {
            innerBroadcast(left, right, destination, ::matmul)
        }
    }

    if (rank <= 2 && other.rank <= 2) {
        val actualThis: NumberNDArray = if (rank == 1) this.toMutable().reshape(1.concat(shape)) else this
        val actualOther = if (other.rank == 1) this.toMutable().reshape(other.shape.concat(1)) else other

        val newStrides = Strides(intArrayOf(shape[0], other.shape[1]))
        val destination = allocateNDArray(newStrides)

        return actualThis.dot(actualOther, destination)
    }

    val outputMatrixShape = intArrayOf(shape[indexAxis(-2)], other.shape[other.indexAxis(-1)])
    val broadcastShape = broadcastShape(shape.copyOfRange(0, rank - 2), other.shape.copyOfRange(0, other.rank - 2))

    val outputShape = IntArray(broadcastShape.size + 2)
    broadcastShape.copyInto(outputShape)
    outputMatrixShape.copyInto(outputShape, broadcastShape.size)

    val outputStrides = Strides(outputShape)
    val outputArray = allocateNDArray(outputStrides)

    val leftWrapShape = unsqueezeFirst(shape, outputShape.size)
    val rightWrapShape = unsqueezeFirst(other.shape, outputShape.size)

    val leftWrapped = this.toMutable(Strides(leftWrapShape))
    val rightWrapped = other.toMutable(Strides(rightWrapShape))

    matmul(leftWrapped, rightWrapped, outputArray)
    return outputArray
}

fun viewHelper(axes: IntArray, strides: Strides): Pair<Int, IntArray> {
    val newOffset = axes.foldIndexed(0) { index, acc, i -> acc + i * strides.strides[index] }
    val newShape = strides.shape.copyOfRange(axes.size, strides.shape.size)

    return newOffset to newShape
}

fun MutableNDArray.reshape(tensorShape: NDArray): MutableNDArray {
    val newShape = IntArray(tensorShape.linearSize) { i -> (tensorShape[i] as Number).toInt() }
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
