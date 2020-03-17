package org.jetbrains.research.kotlin.mpp.inference.tensors

import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.space.*
import scientifik.kmath.structures.*

//TODO: support segments
//TODO: support external and raw data
//TODO: numpy-like multidirectional broadcasting
class Tensor<T : Number>(val name: String?, val data: NDBuffer<T>, val type: DataType?, private val space: TensorRing<T>?) {
    operator fun plus(other: Tensor<T>): Tensor<T>? {
        require(type != DataType.STRING) { "Available only for numeric tensors" }

        val res = space!!.add(data, other.data)
        return Tensor(name, res, type, space)
    }

    infix fun dot(other: Tensor<T>): Tensor<T> {
        require(data.dimension <= 2) { "Not supported for more than 2-dimensional tensors" }

        val resMatrix = with (space!!.matrixContext) { other.data.as2D() dot data.as2D() }
        val newSpace = space.rebuild(newDims = resMatrix.shape)
        return Tensor(name ?: "", resMatrix, type, newSpace)
    }

    fun mapElements(func: (T) -> T): Tensor<T> {
        val newData = BufferNDStructure(DefaultStrides(data.shape), data.buffer.asIterable().map(func).asBuffer())
        return Tensor(name, newData, type, space)
    }

    companion object {
        //TODO: complex, uint32/64 tensors, strings
        fun create(proto: TensorProto): Tensor<*> = when (val type = DataType.fromValue(proto.data_type ?: 0)) {
            DataType.DOUBLE -> Tensor(proto.dims, proto.double_data, type, proto.name, resolveSpace(proto.dims))
            DataType.FLOAT -> Tensor(proto.dims, proto.float_data, type, proto.name, resolveSpace(proto.dims))
            DataType.INT64 -> Tensor(proto.dims, proto.int64_data, type, proto.name, resolveSpace(proto.dims))
            DataType.INT32, DataType.INT8, DataType.UINT8, DataType.UINT16,
            DataType.INT16, DataType.BOOL, DataType.FLOAT16 -> Tensor(proto.dims, proto.int32_data, type, proto.name, resolveSpace(proto.dims))
            else -> throw IllegalArgumentException("Unsupported data type")
        }

        private operator fun <T : Number> invoke(name: String, matrix: Matrix<T>, type: DataType?, space: TensorRing<T>?): Tensor<T> {
            val buffer = matrix.elements().map { it.second }.toList().asBuffer()
            return Tensor(name, BufferNDStructure(DefaultStrides(matrix.shape), buffer as Buffer<T>), type, space)
        }

        //TODO: infer type from graph ValueInfo
        operator fun <T : Number> invoke(dims: List<Long>, value: List<T>, type: DataType?, name: String?, space: TensorRing<T>?): Tensor<T> {
            val data = BufferNDStructure(DefaultStrides(dims.asIntArray().reversedArray()), value.asBuffer())
            return Tensor(name, data, type, space!!)
        }

        inline operator fun <reified T : Number> invoke(value: List<T>, type: DataType?): Tensor<T> {
            val dims = intArrayOf(value.size, 1)
            val data = BufferNDStructure(DefaultStrides(dims), value.asBuffer())
            val space = tryResolveSpace<T>(dims)
            return Tensor(null, data, type, space)
        }
    }
}

