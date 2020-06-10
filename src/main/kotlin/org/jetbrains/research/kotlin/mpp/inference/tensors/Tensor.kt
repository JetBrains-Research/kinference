package org.jetbrains.research.kotlin.mpp.inference.tensors

import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.space.*
import org.jetbrains.research.kotlin.mpp.inference.types.resolveKClass
import scientifik.kmath.linear.GenericMatrixContext
import scientifik.kmath.linear.transpose
import scientifik.kmath.operations.Ring
import scientifik.kmath.structures.*

//TODO: support segments
//TODO: support external and raw data
//TODO: numpy-like multidirectional broadcasting
class Tensor(var name: String?, val data: NDBuffer<Any>, val type: DataType?, val space: TensorRing<Any>?) {
    val elementsList: List<Any>
        get() = data.buffer.asIterable().toList()

    val rank: Int
        get() = data.dimension

    operator fun plus(other: Tensor): Tensor {
        require(type != DataType.STRING) { "Available only for numeric tensors" }

        val result = space!!.combine(data, other.data) { fst, snd -> fst + snd }
        return Tensor(name, result, type, space)
    }

    infix fun dot(other: Tensor): Tensor {
        require(data.dimension <= 2) { "Not supported for more than 2-dimensional tensors" }

        val context = resolveMatrixContext(type!!.resolveKClass()) as GenericMatrixContext<Any, Ring<Any>>
        val resMatrix = with(context) { data.as2D() dot other.data.as2D() }
        val newSpace = space!!.rebuild(newDims = resMatrix.shape)
        return Tensor(name, resMatrix, type, newSpace)
    }


    fun mapElements(func: (Any) -> Any): Tensor {
        val newData = BufferNDStructure(SpaceStrides(data.shape), data.buffer.asIterable().map(func).asBuffer())
        return Tensor(name, newData, type, space)
    }

    operator fun times(other: Tensor): Tensor {
        require(data.shape.contentEquals(other.data.shape))

        val newData = space!!.multiply(this.data, other.data)
        return Tensor(name, newData, type, space)
    }

    fun transpose(): Tensor {
        require(data.dimension <= 2) { "Not supported for more than 2-dimensional tensors" }

        val resMatrix = data.as2D().transpose()
        val newSpace = space?.rebuild(newDims = resMatrix.shape)
        return Tensor(name, resMatrix, type, newSpace)
    }

    fun as2DCollection(): Collection<Tensor> {
        require(data.dimension == 3)

        val blockSize = data.shape[1] * data.shape[2]
        val newShape = intArrayOf(data.shape[1], data.shape[2])
        val newSpace = space!!.rebuild(newShape)
        val newStrides = SpaceStrides(newShape)
        return List(data.shape[0]) { index ->
            val newBuffer = VirtualBuffer(blockSize) { i ->
                val indices = newStrides.index(i)
                val rowNum = indices[0]
                val colNum = indices[1]
                data[index, rowNum, colNum]
            }
            val newStructure = BufferNDStructure(newStrides, newBuffer)
            Tensor(null, newStructure, type, newSpace)
        }
    }

    fun mapIndexed(transform: Ring<Any>.(index: IntArray, Any) -> Any): Tensor {
        val newBuffer = space!!.mapIndexed(data, transform)
        return Tensor(name, newBuffer, type, space)
    }

    // A function that divides the tensor into several parts just like in numpy, where "index" is "axis" in numpy
    fun splitByIndex(parts: Int, index: Int = 0): List<Tensor> {
        require(index in data.shape.indices) { "Index $index out of shape bound: (0, ${data.dimension - 1}" }

        val elementsByIndex = data.shape[index]

        require(elementsByIndex % parts == 0) { "$elementsByIndex is not divisible by $parts" }

        val elementsInChunk = elementsByIndex.div(parts)
        val newShape = data.shape.copyOf()
        newShape[index] = elementsInChunk
        val newStrides = SpaceStrides(newShape)
        val blockSize = newStrides.linearSize
        val newSpace = space!!.rebuild(newShape)
        return List(parts) { num ->
            val newBuffer = VirtualBuffer(blockSize) { i ->
                val indices = newStrides.index(i)
                indices[index] += num * elementsInChunk
                data[indices]
            }
            val newStructure = BufferNDStructure(newStrides, newBuffer)
            Tensor(null, newStructure, type, newSpace)
        }
    }

    fun reshape(shape: IntArray): Tensor {
        val newStrides = SpaceStrides(shape)

        require(data.strides.linearSize == newStrides.linearSize) { "New shape is not compatible with the previous one" }

        val newBuffer = BufferNDStructure(newStrides, data.buffer)
        val newSpace = space!!.rebuild(shape)

        return Tensor(name, newBuffer, type, newSpace)
    }

    fun squeeze(index: Int): Tensor {
        require(data.shape[index] == 1) { "shape[$index] == ${data.shape[index]}, but require 1" }

        val shapeIndices = data.shape.indices.minus(index)
        val newShape = data.shape.slice(shapeIndices).toIntArray()

        return reshape(newShape)
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Tensor) return false

        return type == other.type && data == other.data
    }

    override fun hashCode(): Int {
        var result = data.hashCode()
        result = 31 * result + (type?.hashCode() ?: 0)
        return result
    }

    companion object {
        //TODO: complex, uint32/64 tensors, strings
        fun create(proto: TensorProto): Tensor = when (val type = DataType.fromValue(proto.data_type ?: 0)) {
            DataType.DOUBLE -> Tensor(proto.dims, proto.double_data, type, proto.name, resolveSpace<Double>(proto.dims))
            DataType.FLOAT -> Tensor(proto.dims, proto.float_data, type, proto.name, resolveSpace<Float>(proto.dims))
            DataType.INT64 -> Tensor(proto.dims, proto.int64_data, type, proto.name, resolveSpace<Long>(proto.dims))
            DataType.INT32 -> Tensor(proto.dims, proto.int32_data, type, proto.name, resolveSpace<Int>(proto.dims))
            else -> error("Unsupported data type")
        }

        private operator fun invoke(name: String?, matrix: Matrix<*>, type: DataType?, space: TensorRing<*>?): Tensor {
            val buffer = matrix.elements().map { it.second }.toList().asBuffer()
            return Tensor(name, BufferNDStructure(SpaceStrides(matrix.shape), buffer as Buffer<Any>), type, space as TensorRing<Any>?)
        }

        operator fun invoke(dims: List<Long>, value: List<*>, type: DataType?, name: String?, space: TensorRing<*>?): Tensor {
            val data = BufferNDStructure(SpaceStrides(dims.toIntArray()), value.asBuffer() as Buffer<Any>)
            return Tensor(name, data, type, space as TensorRing<Any>?)
        }

        operator fun invoke(value: List<Any>, type: DataType?): Tensor {
            val dims = intArrayOf(value.size, 1)
            val data = BufferNDStructure(SpaceStrides(dims), value.asBuffer())
            val space = resolveSpaceWithKClass(type!!.resolveKClass(), dims)
            return Tensor(null, data, type, space as TensorRing<Any>)
        }
    }
}

