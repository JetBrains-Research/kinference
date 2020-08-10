package org.jetbrains.research.kotlin.inference.data.ndarray

import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.data.tensors.broadcast
import org.jetbrains.research.kotlin.inference.data.tensors.broadcastMatrixElementsShape
import org.jetbrains.research.kotlin.inference.extensions.functional.PrimitiveArrayFunction
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*
import org.jetbrains.research.kotlin.inference.extensions.primitives.matrixDot
import org.jetbrains.research.kotlin.inference.extensions.primitives.reversed
import org.jetbrains.research.kotlin.inference.extensions.primitives.toIntArray
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.onnx.TensorProto.DataType
import org.jetbrains.research.kotlin.inference.types.TensorInfo
import org.jetbrains.research.kotlin.inference.types.TensorShape
import kotlin.math.abs

abstract class NDArray<T> protected constructor(val array: T, val strides: Strides, val type: DataType) {
    val rank: Int
        get() = strides.shape.size

    val linearSize: Int
        get() = strides.linearSize

    val shape: IntArray
        get() = strides.shape

    val rows: Array<NDArray<T>>
        get() {
            val rowLength: Int = linearSize / shape[0]
            val dims = shape.copyOfRange(1, rank)

            return Array(shape[0]) { row -> sliceRow(rowLength, row * rowLength, dims) }
        }

    fun isScalar(): Boolean = shape.isEmpty()

    abstract operator fun get(i: Int): Any
    abstract operator fun set(i: Int, value: Any)
    abstract operator fun get(indices: IntArray): Any

    // TODO add similar method with IntRange and Arrays.copy
    abstract fun appendToLateInitArray(array: LateInitArray, range: IntProgression, offset: Int)

    abstract fun clone(newStrides: Strides = strides): NDArray<T>
    abstract fun place(startOffset: Int, block: Any?, startIndex: Int, endIndex: Int)
    abstract fun placeAll(startOffset: Int, block: Any?)

    abstract fun plus(other: NDArray<T>, copy: Boolean = true): NDArray<T>
    operator fun plus(other: NDArray<T>) = plus(other, true)

    abstract fun minus(other: NDArray<T>, copy: Boolean = true): NDArray<T>
    operator fun minus(other: NDArray<T>): NDArray<T> = minus(other, true)

    abstract fun times(other: NDArray<T>, copy: Boolean = true): NDArray<T>
    operator fun times(other: NDArray<T>) = times(other, true)

    abstract fun div(other: NDArray<T>, copy: Boolean = true): NDArray<T>
    operator fun div(other: NDArray<T>): NDArray<T> = div(other, true)

    abstract fun mapElements(func: PrimitiveArrayFunction, copy: Boolean = true): NDArray<T>
    abstract fun clean(): Unit
    abstract fun slice(sliceLength: Int, start: Int): Any


    fun indexAxis(axis: Int): Int {
        return if (axis < 0) rank + axis else axis
    }

    infix fun matmul(other: NDArray<T>): NDArray<T> {
        require(!this.isScalar() && !other.isScalar()) { "Matmul operation is not available for scalar tensors" }
        if (rank <= 2 && other.rank <= 2) {
            val actualThis = if (rank == 1) this.reshape(intArrayOf(1, *shape)) else this
            val actualOther = if (other.rank == 1) this.reshape(intArrayOf(*other.shape, 1)) else other
            return actualThis.matrixDot(actualOther)
        }

        val (fstShape, sndShape) = broadcastMatrixElementsShape(shape, other.shape)
        val thisMatrices = this.broadcast(fstShape, asMatrixStack = true).as2DList()
        val otherMatrices = other.broadcast(sndShape, asMatrixStack = true).as2DList()

        val resMatrices = thisMatrices.mapIndexed { i, tensor ->
            tensor.matrixDot(otherMatrices[i])
        }

        val lastDims = resMatrices.first().shape

        val shape = shape.copyOf(rank - 2) + lastDims
        return resMatrices.concatenate(0).reshape(shape)
    }

    fun row(row: Int): NDArray<T> {
        val rowLength: Int = linearSize / shape[0]
        val start = row * rowLength
        val dims = shape.copyOfRange(1, rank)

        return sliceRow(rowLength, start, dims)
    }

    @Suppress("UNCHECKED_CAST")
    private fun sliceRow(rowLength: Int, start: Int, dims: IntArray): NDArray<T> {
        val row = slice(rowLength, start)
        return NDArray(row, type, dims) as NDArray<T>
    }

    @Suppress("UNCHECKED_CAST")
    fun repeatRow(times: Int): NDArray<T> {
        require(shape[0] == 1) { "First dimension should be 1" }
        val newShape = shape.copyOf().apply { set(0, times) }

        val result = allocateNDArray(type, Strides(newShape))
        for (i in 0 until times) {
            result.placeAll(i * linearSize, array)
        }

        return result as NDArray<T>
    }

    fun transpose(permutations: List<Number>? = null): NDArray<T> {
        if (rank == 2) return this.matrixTranspose()

        require(permutations.isNullOrEmpty() || permutations.size == rank) { "Axes permutations list size should match the number of axes" }
        val actualPerm = if (permutations.isNullOrEmpty()) shape.indices.reversed() else permutations.toIntArray()

        return this.transpose(actualPerm)
    }

    fun reshape(shape: IntArray): NDArray<T> {
        val newStrides = Strides(shape)
        require(linearSize == newStrides.linearSize) { "New shape is not compatible with the previous one" }

        return clone(newStrides)
    }

    fun squeeze(vararg axes: Int): NDArray<T> {
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

    fun unsqueeze(vararg axes: Int): NDArray<T> {
        val actualAxes = axes.map { indexAxis(it) }.sorted()
        val newShape = shape.toMutableList()
        for (axis in actualAxes) {
            newShape.add(axis, 1)
        }
        return reshape(newShape.toIntArray())
    }

    fun slice(starts: IntArray, ends: IntArray, steps: IntArray): NDArray<T> {
        val newShape = IntArray(shape.size) {
            val length = abs(ends[it] - starts[it])
            val rest = length % abs(steps[it])
            (length / abs(steps[it])) + if (rest != 0) 1 else 0
        }

        val newStrides = Strides(newShape)
        val newArray = createLateInitArray(type, newStrides)

        slice(newArray, 0, 0, shape, starts, ends, steps)

        return createNDArrayFromLateInitArray(type, newArray, newStrides) as NDArray<T>
    }

    private fun slice(dest: LateInitArray, offset: Int, axis: Int, shape: IntArray, starts: IntArray, ends: IntArray, steps: IntArray) {
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

    // TODO: better equals
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as NDArray<T>

        if (array != other.array) return false

        return true
    }

    override fun hashCode(): Int {
        return array.hashCode()
    }

    fun asTensor(name: String? = null) = Tensor(this as NDArray<Any>, TensorInfo(name ?: "", type, TensorShape(this.shape)))

    companion object {
        //TODO: complex, uint32/64 tensors
        fun create(proto: TensorProto): NDArray<out Any> {
            if (proto.dims.isNullOrEmpty()) return createScalar(proto)

            return when (val type = DataType.fromValue(proto.data_type ?: 0)) {
                DataType.DOUBLE -> DoubleNDArray(proto.double_data.toDoubleArray(), Strides(proto.dims.toIntArray()))
                DataType.FLOAT -> FloatNDArray(proto.float_data.toFloatArray(), Strides(proto.dims.toIntArray()))
                DataType.INT64 -> LongNDArray(proto.int64_data.toLongArray(), Strides(proto.dims.toIntArray()))
                DataType.INT32 -> IntNDArray(proto.int32_data.toIntArray(), Strides(proto.dims.toIntArray()))
                //DataType.STRING -> Tensor(proto.string_data.map { it.utf8() }, type, proto.dims.toIntArray(), proto.name)
                else -> error("Unsupported data type $type")
            }
        }

        operator fun <T> invoke(dims: List<Long>, value: List<*>, type: DataType): NDArray<T> {
            val data = createArray(type, value.size) { i -> value[i]!! }
            return NDArray(data, type, dims.toIntArray()) as NDArray<T>
        }


        operator fun <T> invoke(value: T, type: DataType, dims: IntArray = IntArray(0)): NDArray<T> {
            return NDArray(value, type, Strides(dims))
        }

        operator fun <T> invoke(value: T, type: DataType, strides: Strides): NDArray<T> {
            return when (type) {
                DataType.DOUBLE -> DoubleNDArray(value as DoubleArray, strides)
                DataType.FLOAT -> FloatNDArray(value as FloatArray, strides)
                DataType.INT64 -> LongNDArray(value as LongArray, strides)
                DataType.INT32 -> IntNDArray(value as IntArray, strides)
                //DataType.STRING -> TensorData(proto.string_data.map { it.utf8() }, type, proto.dims.toIntArray(), proto.name)
                else -> error("Unsupported data type $type")
            } as NDArray<T>
        }

        operator fun invoke(value: List<*>, type: DataType): NDArray<Any> {
            val dims = intArrayOf(value.size)
            val data = createArray(type, value.size) { i -> value[i] }
            return NDArray(data, type, dims)
        }

        private fun createScalar(proto: TensorProto): NDArray<out Any> {
            val type = DataType.fromValue(proto.data_type ?: 0)
            val array = when (type) {
                DataType.DOUBLE -> proto.double_data
                DataType.FLOAT -> proto.float_data
                DataType.INT64 -> proto.int64_data
                DataType.INT32 -> proto.int32_data
                DataType.BOOL -> proto.int32_data.map { it != 0 }
                else -> error("Unsupported data type")
            }

            return if (array.isEmpty()) {
                when (type) {
                    DataType.DOUBLE -> DoubleNDArray(doubleArrayOf(proto.raw_data!!.asByteBuffer().double))
                    DataType.FLOAT -> FloatNDArray(floatArrayOf(proto.raw_data!!.asByteBuffer().float))
                    DataType.INT64 -> LongNDArray(longArrayOf(proto.raw_data!!.asByteBuffer().long))
                    DataType.INT32 -> IntNDArray(intArrayOf(proto.raw_data!!.asByteBuffer().int))
                    DataType.BOOL -> BooleanNDArray(booleanArrayOf(proto.raw_data!!.asByteBuffer().int != 0))
                    else -> error("Unsupported data type")
                }
            } else NDArray(array[0], type, IntArray(0))
        }
    }
}
