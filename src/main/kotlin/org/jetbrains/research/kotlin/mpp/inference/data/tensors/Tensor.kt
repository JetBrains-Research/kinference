package org.jetbrains.research.kotlin.mpp.inference.data.tensors

import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.types.*
import scientifik.kmath.linear.GenericMatrixContext
import scientifik.kmath.operations.Ring
import scientifik.kmath.structures.*
import kotlin.math.min

//TODO: support segments
//TODO: support external and raw data
@Suppress("UNCHECKED_CAST")
class Tensor(val data: NDBuffer<Any>, info: TensorInfo) : BaseTensor(info) {
    constructor(name: String?, data: NDBuffer<Any>, type: DataType) : this(data, TensorInfo(name ?: "", type, TensorShape(data.shape)))

    val elementsList: List<Any>
        get() = data.buffer.asIterable().toList()

    val rank: Int
        get() = data.dimension

    override fun clone(newName: String) = Tensor(newName, data, info.type)

    override fun plus(other: BaseTensor): BaseTensor {
        return when (other) {
            is Tensor -> {
                require(info.type != DataType.STRING) { "Available only for numeric tensors" }
                if (!data.shape.contentEquals(other.data.shape)) {
                    return elementWiseWithBroadcast(other) { fst, snd -> add(fst as Number, snd as Number) }
                }

                data as BufferNDStructure<Number>; other.data as BufferNDStructure<Number>
                val result = data.ndCombine(other.data) { fst, snd -> add(fst, snd) }
                Tensor("output", result as BufferNDStructure<Any>, info.type)
            }
            is ScalarTensor -> this.mapElements { add(it as Number, other.value as Number) }
            else -> error("Unsupported tensor type")
        }
    }

    fun indexAxis(axis: Int): Int {
        return if (axis < 0) data.shape.size + axis else axis
    }

    fun row(row: Int): Tensor {
        val rowLength: Int = data.strides.linearSize / data.shape[0]
        val start = row * rowLength
        val rowData = data.buffer.asIterable().toList().subList(start, start + rowLength)
        val dims = data.shape.copyOf().drop(1).toIntArray()

        val buffer = BufferNDStructure(TensorStrides(dims), rowData.asBuffer())
        return Tensor("row", buffer, info.type)
    }

    fun rows(): List<Tensor> = List(data.shape[0]) { i -> row(i) }

    fun repeatRow(times: Int): Tensor {
        require(data.shape[0] == 1) { "First dimension should be 1" }
        val resultBuffer = List(times) { data.buffer.asIterable() }.flatten().asBuffer()
        val newShape = data.shape.copyOf().apply { set(0, times) }
        return Tensor("rows", BufferNDStructure(TensorStrides(newShape), resultBuffer), info.type)
    }

    override fun matmul(other: BaseTensor): BaseTensor {
        return when (other) {
            is Tensor -> {
                val context = resolveMatrixContext(info.type.resolveKClass()) as GenericMatrixContext<Any, Ring<Any>>

                if (data.dimension <= 2 && other.data.dimension <= 2) {
                    val matrix = with(context) { data.as2D() dot other.data.as2D() }
                    return Tensor("result", matrix, info.type)
                }

                val thisMatrices = this.as2DList()
                val otherMatrices = other.as2DList()
                val resMatrices = thisMatrices.mapIndexed { i, tensor ->
                    with(context) { tensor.data.as2D() dot otherMatrices[i].data.as2D() }
                }.map { matrix ->
                    val buffer = matrix.elements().map { it.second }.toList().asBuffer()
                    val nd = BufferNDStructure(TensorStrides(matrix.shape), buffer)
                    Tensor("out", nd, info.type)
                }
                val shape = data.shape.zip(other.data.shape).map { min(it.first, it.second) }.toIntArray()
                resMatrices.concatenate(0).reshape(shape)
            }
            is ScalarTensor -> this * other
            else -> error("Unsupported tensor type")
        }
    }

    fun mapElements(type: DataType = info.type, func: (Any) -> Any): Tensor {
        val newData = BufferNDStructure(TensorStrides(data.shape), data.buffer.asIterable().map(func).asBuffer())
        return Tensor(info.name, newData, type)
    }

    override fun times(other: BaseTensor): BaseTensor {
        return when (other) {
            is Tensor -> {
                require(data.shape.contentEquals(other.data.shape))
                require(info.type != DataType.STRING) { "Available only for numeric tensors" }
                data as BufferNDStructure<Number>; other.data as BufferNDStructure<Number>

                val result = data.ndCombine(other.data) { fst, snd -> times(fst, snd) }
                Tensor(info.name, result as BufferNDStructure<Any>, info.type)
            }
            is ScalarTensor -> this.mapElements { times(it as Number, other.value as Number) }
            else -> error("Unsupported tensor type")
        }
    }

    fun transpose(permutations: List<Long>? = null): Tensor {
        require(permutations.isNullOrEmpty() || permutations.size == rank) { "Axes permutations list size should match the number of axes" }
        val actualPerm = if (permutations.isNullOrEmpty()) data.shape.indices.reversed() else permutations.toIntArray()

        val newShape = IntArray(rank)
        for ((i, axis) in actualPerm.withIndex()) {
            newShape[i] = data.shape[axis]
        }
        val newStrides = TensorStrides(newShape)

        val newBuffer = MutableBufferNDStructure<Any?>(newStrides, MutableBuffer.boxing(newStrides.linearSize) { null })
        for (i in 0 until data.strides.linearSize) {
            val indices = data.strides.index(i)
            val newIndices = IntArray(indices.size)
            for ((id, axis) in actualPerm.withIndex()) {
                newIndices[id] = indices[axis]
            }
            newBuffer[newIndices] = data[indices]
        }
        return Tensor(info.name, newBuffer as NDBuffer<Any>, info.type)
    }

    fun splitWithAxis(split: IntArray, axis: Int = 0, keepDims: Boolean = true): List<Tensor> {
        return List(split.size) { num ->
            val newShape = data.shape.copyOf().toMutableList()
            if (!keepDims) newShape.removeAt(axis) else newShape[axis] = split[num]
            val newStrides = TensorStrides(newShape.toIntArray())
            val blockSize = newStrides.linearSize
            val newBuffer = ListBuffer(blockSize) { i ->
                val indices = newStrides.index(i)
                indices[axis] += num * (split.getOrNull(num - 1) ?: 0)
                data[indices]
            }
            val newStructure = BufferNDStructure(newStrides, newBuffer)
            Tensor(null, newStructure, info.type)
        }
    }

    fun reshape(shape: IntArray): Tensor {
        val newStrides = TensorStrides(shape)
        require(data.strides.linearSize == newStrides.linearSize) { "New shape is not compatible with the previous one" }

        val newBuffer = BufferNDStructure(newStrides, data.buffer)
        return Tensor(info.name, newBuffer, info.type)
    }

    fun reshape(tensorShape: Tensor): Tensor {
        val requestedShape = (tensorShape.elementsList as List<Long>).toIntArray()
        require(requestedShape.count { it == -1 } <= 1) { "At most one dimension of the new shape can be -1" }

        val newShape = requestedShape.toMutableList()
        for ((i, axisShape) in requestedShape.withIndex()) {
            if (axisShape == 0) newShape[i] = data.shape[i]
        }

        val negativeIdx = newShape.indexOf(-1)
        if (negativeIdx != -1) {
            val elementsCount = newShape.filter { it != -1 }.reduce(Int::times)
            newShape[negativeIdx] = data.shape.reduce(Int::times) / elementsCount
        }

        return reshape(newShape.toIntArray())
    }

    fun squeeze(vararg axes: Int): Tensor {
        val actualAxes = if (axes.isNotEmpty()) {
            axes.map { indexAxis(it) }
        } else {
            data.shape.withIndex().filter { it.value == 1 }.map { it.index }
        }
        require(actualAxes.all { data.shape[it] == 1 })

        val shapeIndices = data.shape.indices - actualAxes
        val newShape = data.shape.slice(shapeIndices).toIntArray()

        return reshape(newShape)
    }

    // TODO: better equals
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Tensor

        if (data != other.data) return false

        return true
    }

    override fun hashCode(): Int {
        return data.hashCode()
    }

    companion object {
        //TODO: complex, uint32/64 tensors
        fun create(proto: TensorProto): Tensor = when (val type = DataType.fromValue(proto.data_type ?: 0)) {
            DataType.DOUBLE -> Tensor(proto.dims, proto.double_data, type, proto.name)
            DataType.FLOAT -> Tensor(proto.dims, proto.float_data, type, proto.name)
            DataType.INT64 -> Tensor(proto.dims, proto.int64_data, type, proto.name)
            DataType.INT32 -> Tensor(proto.dims, proto.int32_data, type, proto.name)
            DataType.STRING -> Tensor(proto.dims, proto.string_data.map { it.utf8() }, type, proto.name)
            else -> error("Unsupported data type")
        }

        private operator fun invoke(name: String?, matrix: Matrix<*>, type: DataType): Tensor {
            val buffer = matrix.elements().map { it.second }.toList().asBuffer()
            return Tensor(name, BufferNDStructure(TensorStrides(matrix.shape), buffer as Buffer<Any>), type)
        }

        operator fun invoke(dims: List<Long>, value: List<*>, type: DataType, name: String?): Tensor {
            val data = BufferNDStructure(TensorStrides(dims.toIntArray()), value.asBuffer() as Buffer<Any>)
            return Tensor(name, data, type)
        }

        operator fun invoke(value: List<Any>, type: DataType): Tensor {
            val dims = intArrayOf(value.size)
            val data = BufferNDStructure(TensorStrides(dims), value.asBuffer())
            return Tensor("out", data, type)
        }
    }
}

