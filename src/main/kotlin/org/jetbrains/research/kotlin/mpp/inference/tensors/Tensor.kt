package org.jetbrains.research.kotlin.mpp.inference.tensors

import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.types.resolveKClass
import scientifik.kmath.linear.GenericMatrixContext
import scientifik.kmath.operations.Ring
import scientifik.kmath.structures.*

//TODO: support segments
//TODO: support external and raw data
//TODO: numpy-like multidirectional broadcasting
@Suppress("UNCHECKED_CAST")
data class Tensor(val name: String?, val data: NDBuffer<Any>, val type: DataType) {
    val elementsList: List<Any>
        get() = data.buffer.asIterable().toList()

    val rank: Int
        get() = data.dimension

    operator fun plus(other: Tensor): Tensor {
        require(type != DataType.STRING) { "Available only for numeric tensors" }
        data as BufferNDStructure<Number>; other.data as BufferNDStructure<Number>

        val result = data.ndCombine(other.data) { fst, snd -> add(fst, snd) }
        return Tensor(name, result as BufferNDStructure<Any>, type)
    }

    fun row(row: Int): Tensor {
        val rowLength: Int = data.strides.linearSize / data.shape[0]
        val start = row * rowLength
        val rowData = data.buffer.asIterable().toList().subList(start, start + rowLength)
        val dims = if (data.shape.size == 2) {
            intArrayOf(data.shape[1])
        } else {
            data.shape.copyOfRange(1, data.shape.size)
        }
        val buffer = BufferNDStructure(TensorStrides(dims), rowData.asBuffer())
        return Tensor("row", buffer, type)
    }

    infix fun dot(other: Tensor): Tensor {
        require(data.dimension <= 2) { "Not supported for more than 2-dimensional tensors" }

        val context = resolveMatrixContext(type.resolveKClass()) as GenericMatrixContext<Any, Ring<Any>>
        val resMatrix = with(context) { data.as2D() dot other.data.as2D() }
        return Tensor(name, resMatrix, type)
    }


    fun mapElements(func: (Any) -> Any): Tensor {
        val newData = BufferNDStructure(TensorStrides(data.shape), data.buffer.asIterable().map(func).asBuffer())
        return Tensor(name, newData, type)
    }

    operator fun times(other: Tensor): Tensor {
        require(data.shape.contentEquals(other.data.shape))
        require(type != DataType.STRING) { "Available only for numeric tensors" }
        data as BufferNDStructure<Number>; other.data as BufferNDStructure<Number>

        val result = data.ndCombine(other.data) { fst, snd -> times(fst, snd) }
        return Tensor(name, result as BufferNDStructure<Any>, type)
    }

    fun transpose(perm: List<Long>? = null): Tensor {
        val actualPerm = if (perm.isNullOrEmpty()) data.shape.indices.reversed() else perm.toIntArray()
        val newShape = IntArray(data.shape.size)
        for ((i, axis)  in actualPerm.withIndex()) {
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
        return Tensor(name, newBuffer as NDBuffer<Any>, type)
    }

    fun splitWithAxis(split: IntArray, axis: Int = 0): List<Tensor> {
        return List(split.size) { num ->
            val newShape = data.shape.copyOf()
            newShape[axis] = split[num]
            val newStrides = TensorStrides(newShape)
            val blockSize = newStrides.linearSize
            val newBuffer = ListBuffer(blockSize) { i ->
                val indices = newStrides.index(i)
                indices[axis] += num * (split.getOrNull(num - 1) ?: 0)
                data[indices]
            }
            val newStructure = BufferNDStructure(newStrides, newBuffer)
            Tensor(null, newStructure, type)
        }
    }

    fun reshape(shape: IntArray): Tensor {
        val newStrides = TensorStrides(shape)

        require(data.strides.linearSize == newStrides.linearSize) { "New shape is not compatible with the previous one" }

        val newBuffer = BufferNDStructure(newStrides, data.buffer)

        return Tensor(name, newBuffer, type)
    }

    fun reshape(tensorShape: Tensor): Tensor {
        val shape = tensorShape.elementsList as List<Long>
        val newShape = shape.toMutableList()
        for ((i, axisShape) in shape.withIndex()) {
            if (axisShape == 0L) newShape[i] = data.shape[i].toLong()
        }

        val negIdx = newShape.indexOf(-1L)
        if (negIdx != -1) {
            val elementsCount = newShape.filter { it != -1L }.reduce(Long::times)
            newShape[negIdx] = data.shape.reduce(Int::times).toLong() / elementsCount
        }

        return reshape(newShape.toIntArray())
    }

    fun squeeze(index: Int): Tensor {
        require(data.shape[index] == 1) { "shape[$index] == ${data.shape[index]}, but require 1" }

        val shapeIndices = data.shape.indices.minus(index)
        val newShape = data.shape.slice(shapeIndices).toIntArray()

        return reshape(newShape)
    }

    companion object {
        //TODO: complex, uint32/64 tensors, strings
        fun create(proto: TensorProto): Tensor = when (val type = DataType.fromValue(proto.data_type ?: 0)) {
            DataType.DOUBLE -> Tensor(proto.dims, proto.double_data, type, proto.name)
            DataType.FLOAT -> Tensor(proto.dims, proto.float_data, type, proto.name)
            DataType.INT64 -> Tensor(proto.dims, proto.int64_data, type, proto.name)
            DataType.INT32 -> Tensor(proto.dims, proto.int32_data, type, proto.name)
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
            val dims = intArrayOf(value.size, 1)
            val data = BufferNDStructure(TensorStrides(dims), value.asBuffer())
            return Tensor(null, data, type)
        }
    }
}

