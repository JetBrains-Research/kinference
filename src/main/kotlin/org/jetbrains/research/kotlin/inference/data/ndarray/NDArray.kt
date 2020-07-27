package org.jetbrains.research.kotlin.inference.data.ndarray

import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.inference.data.tensors.*
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.types.TensorInfo
import org.jetbrains.research.kotlin.inference.types.TensorShape

abstract class NDArray protected constructor(val array: Any, val strides: Strides, val type: DataType) {
    val rank: Int
        get() = strides.shape.size

    val linearSize: Int
        get() = strides.linearSize

    val shape: IntArray
        get() = strides.shape

    val rows: Array<NDArray>
        get() {
            val rowLength: Int = linearSize / shape[0]
            val dims = shape.copyOfRange(1, rank)

            return Array(shape[0]) { row -> sliceRow(rowLength, row * rowLength, dims) }
        }

    fun isScalar(): Boolean = shape.isEmpty()

    abstract operator fun get(i: Int): Any
    abstract operator fun get(indices: IntArray): Any

    abstract fun clone(newStrides: Strides = strides): NDArray
    abstract fun placeAll(startOffset: Int, block: Any)

    abstract operator fun plus(other: NDArray): NDArray
    abstract operator fun minus(other: NDArray): NDArray
    abstract operator fun times(other: NDArray): NDArray
    abstract operator fun div(other: NDArray): NDArray

    fun indexAxis(axis: Int): Int {
        return if (axis < 0) rank + axis else axis
    }

    infix fun matmul(other: NDArray): NDArray {
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

    fun row(row: Int): NDArray {
        val rowLength: Int = linearSize / shape[0]
        val start = row * rowLength
        val dims = shape.copyOfRange(1, rank)

        return sliceRow(rowLength, start, dims)
    }

    private fun sliceRow(rowLength: Int, start: Int, dims: IntArray): NDArray {
        val row = slice(rowLength, start)
        return NDArray(row, type, dims)
    }

    private fun slice(sliceLength: Int, start: Int): Any {
        return createArray(type, sliceLength) { i -> this[start + i] }
    }

    fun repeatRow(times: Int): NDArray {
        require(shape[0] == 1) { "First dimension should be 1" }
        val newShape = shape.copyOf().apply { set(0, times) }

        val result = allocateNDArray(type, Strides(newShape))
        for (i in 0 until times) {
            result.placeAll(i * linearSize, array)
        }

        return result
    }

    fun mapElements(func: (Any) -> Any): NDArray {
        val buffer = createArray(type, linearSize) { func(this[it]) }

        return NDArray(buffer, type, shape)
    }

    fun transpose(permutations: List<Long>? = null): NDArray {
        if (rank == 2) return this.matrixTranspose()

        require(permutations.isNullOrEmpty() || permutations.size == rank) { "Axes permutations list size should match the number of axes" }
        val actualPerm = if (permutations.isNullOrEmpty()) shape.indices.reversed() else permutations.toIntArray()

        return this.transpose(actualPerm)
    }

    fun reshape(shape: IntArray): NDArray {
        val newStrides = Strides(shape)
        require(linearSize == newStrides.linearSize) { "New shape is not compatible with the previous one" }

        return clone(newStrides)
    }

    fun squeeze(vararg axes: Int): NDArray {
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

    // TODO: better equals
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as NDArray

        if (array != other.array) return false

        return true
    }

    override fun hashCode(): Int {
        return array.hashCode()
    }

    fun asTensor(name: String? = null) = Tensor(this, TensorInfo(name ?: "", type, TensorShape(this.shape)))

    companion object {
        //TODO: complex, uint32/64 tensors
        fun create(proto: TensorProto): NDArray {
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

        operator fun invoke(dims: List<Long>, value: List<*>, type: DataType): NDArray {
            val data = createArray(type, value.size) { i -> value[i]!! }
            return NDArray(data, type, dims.toIntArray())
        }


        operator fun invoke(value: Any, type: DataType, dims: IntArray = IntArray(0)): NDArray {
            return NDArray(value, type, Strides(dims))
        }

        operator fun invoke(value: Any, type: DataType, strides: Strides): NDArray {
            return when (type) {
                DataType.DOUBLE -> DoubleNDArray(value as DoubleArray, strides)
                DataType.FLOAT -> FloatNDArray(value as FloatArray, strides)
                DataType.INT64 -> LongNDArray(value as LongArray, strides)
                DataType.INT32 -> IntNDArray(value as IntArray, strides)
                //DataType.STRING -> TensorData(proto.string_data.map { it.utf8() }, type, proto.dims.toIntArray(), proto.name)
                else -> error("Unsupported data type $type")
            }
        }

        operator fun invoke(value: List<Any>, type: DataType): NDArray {
            val dims = intArrayOf(value.size)
            val data = createArray(type, value.size) { i -> value[i] }
            return NDArray(data, type, dims)
        }

        private fun createScalar(proto: TensorProto): NDArray {
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
