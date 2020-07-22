package org.jetbrains.research.kotlin.mpp.inference.data.tensors

import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.*
import org.jetbrains.research.kotlin.mpp.inference.types.TensorInfo
import org.jetbrains.research.kotlin.mpp.inference.types.TensorShape
import scientifik.kmath.linear.BufferMatrix
import scientifik.kmath.structures.*

//TODO: support segments
//TODO: support external and raw data
@Suppress("UNCHECKED_CAST")
class Tensor(val data: NDBuffer<Any>, info: TensorInfo) : BaseTensor(info) {
    constructor(name: String?, data: NDBuffer<Any>, type: DataType) : this(data, TensorInfo(name ?: "", type, TensorShape(data.shape)))

    val rank: Int
        get() = data.dimension

    val rows: Array<Tensor>
        get() {
            val rowLength: Int = data.strides.linearSize / data.shape[0]
            val dims = data.shape.copyOfRange(1, rank)

            return Array(data.shape[0]) { row -> sliceRow(rowLength, row * rowLength, dims)}
        }

    override fun clone(newName: String) = Tensor(newName, data, info.type)

    override fun plus(other: BaseTensor): BaseTensor {
        return when (other) {
            is Tensor -> {
                require(info.type != DataType.STRING) { "Available only for numeric tensors" }
                if (!data.shape.contentEquals(other.data.shape)) {
                    return elementWiseWithBroadcast(other) { fst, snd -> add(fst as Number, snd as Number) }
                }
                data as BufferNDStructure<Number>; other.data as BufferNDStructure<Number>

                val buffer = createBuffer(info.type, data.strides.linearSize) {
                    add(data.buffer[it], other.data.buffer[it])
                }

                Tensor("output", BufferNDStructure(data.strides, buffer) as NDBuffer<Any>, info.type)
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
        val dims = data.shape.copyOfRange(1, rank)

        return sliceRow(rowLength, start, dims)
    }

    private fun sliceRow(rowLength: Int, start: Int, dims: IntArray): Tensor {
        val rowData = createBuffer(info.type, rowLength) { i -> data.buffer[start + i] }
        val buffer = BufferNDStructure(TensorStrides(dims), rowData)
        return Tensor("row", buffer, info.type)
    }

    fun repeatRow(times: Int): Tensor {
        require(data.shape[0] == 1) { "First dimension should be 1" }
        val result = allocateMutableBuffer(info.type, data.buffer.size * times)
        for (i in 0 until times) {
            result.placeAll(data.buffer, i * data.buffer.size)
        }
        val newShape = data.shape.copyOf().apply { set(0, times) }
        return Tensor("rows", BufferNDStructure(TensorStrides(newShape), result), info.type)
    }

    override fun matmul(other: BaseTensor): BaseTensor {
        other as Tensor

        if (data.dimension <= 2 && other.data.dimension <= 2) {
            val actualThis = if (data.dimension == 1) this.reshape(intArrayOf(1, *data.shape)) else this
            val actualOther = if (other.data.dimension == 1) this.reshape(intArrayOf(*other.data.shape, 1)) else other
            val matrix = BufferMatrix(actualThis.data.shape[0], actualThis.data.shape[1], actualThis.data.buffer as Buffer<out Float>)
                .dot(BufferMatrix(actualOther.data.shape[0], actualOther.data.shape[1], actualOther.data.buffer as Buffer<out Float>))
            return Tensor("result", matrix, info.type)
        }

        val (fstShape, sndShape) = broadcastMatrixElementsShape(data.shape, other.data.shape)
        val thisMatrices = this.broadcast(fstShape, asMatrixStack = true).as2DList()
        val otherMatrices = other.broadcast(sndShape, asMatrixStack = true).as2DList()

        val resMatrices = thisMatrices.mapIndexed { i, tensor ->
            BufferMatrix(tensor.data.shape[0], tensor.data.shape[1], tensor.data.buffer as Buffer<out Float>)
                .dot(BufferMatrix(otherMatrices[i].data.shape[0], otherMatrices[i].data.shape[1], otherMatrices[i].data.buffer as Buffer<out Float>))
        }.map { matrix -> Tensor("out", matrix, info.type) }

        val lastDims = resMatrices.first().data.shape

        val shape = data.shape.copyOf(rank - 2) + lastDims
        return resMatrices.concatenate(0).reshape(shape)
    }

    fun mapElements(type: DataType = info.type, func: (Any) -> Any): Tensor {
        val buffer = createBuffer(type, data.strides.linearSize) { func(data.buffer[it]) }
        return Tensor(info.name, BufferNDStructure(data.strides, buffer), type)
    }

    override fun times(other: BaseTensor): BaseTensor {
        return when (other) {
            is Tensor -> {
                require(data.shape.contentEquals(other.data.shape))
                require(info.type != DataType.STRING) { "Available only for numeric tensors" }
                data as BufferNDStructure<Number>; other.data as BufferNDStructure<Number>

                val buffer = createBuffer(info.type, data.strides.linearSize) {
                    times(data.buffer[it], other.data.buffer[it])
                }

                Tensor(info.name, BufferNDStructure(data.strides, buffer) as NDBuffer<Any>, info.type)
            }
            is ScalarTensor -> this.mapElements { times(it as Number, other.value as Number) }
            else -> error("Unsupported tensor type")
        }
    }

    /*fun transpose(permutations: List<Long>? = null): Tensor {
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
    }*/

    fun transpose(permutations: List<Long>? = null): Tensor {
        require(permutations.isNullOrEmpty() || permutations.size == rank) { "Axes permutations list size should match the number of axes" }
        val actualPerm = if (permutations.isNullOrEmpty()) data.shape.indices.reversed() else permutations.toIntArray()

        val newShape = IntArray(rank)
        for ((i, axis) in actualPerm.withIndex()) {
            newShape[i] = data.shape[axis]
        }
        val newStrides = TensorStrides(newShape)

        val newBuffer = createBuffer(info.type, newStrides.linearSize) { i ->
            val indices = newStrides.index(i)
            val newIndices = IntArray(indices.size)
            for ((id, axis) in actualPerm.withIndex()) {
                newIndices[axis] = indices[id]
            }
            data[newIndices]
        }

        val newStructure = BufferNDStructure(newStrides, newBuffer)
        return Tensor(info.name, newStructure, info.type)
    }

    fun splitWithAxis(split: IntArray, axis: Int = 0, keepDims: Boolean = true): List<Tensor> {
        return List(split.size) { num ->
            val newShape = data.shape.copyOf().apply { set(axis, split[num]) }
            val newStrides = TensorStrides(newShape)
            val factor = num * (split.getOrNull(num - 1) ?: 0)
            val newBuffer = createBuffer(info.type, newStrides.linearSize) { i ->
                val indices = newStrides.index(i)
                indices[axis] += factor
                data[indices]
            }

            val newStructure = BufferNDStructure(newStrides, newBuffer)
            val ans = Tensor(null, newStructure, info.type)
            if (!keepDims) ans.squeeze(axis) else ans
        }
    }

    fun gather(indices: Tensor, axis: Int = 0): Tensor {
        val addedShape = data.shape.toMutableList().also { it.removeAt(axis) }
        val newShape = addedShape.toMutableList().also { it.addAll(axis, indices.data.shape.toList()) }
        val newStrides = TensorStrides(newShape.toIntArray())

        val newBuffer = createBuffer(info.type, newStrides.linearSize) { i ->
            val current = newStrides.index(i)
            val indicesIndices = current.sliceArray(axis until indices.data.shape.size + axis)
            val gatherIndices = (current.take(axis) + current.takeLast(data.shape.size - 1 - axis)).toMutableList().also { it.add(axis, (indices.data[indicesIndices] as Long).toInt()) }
            val positiveGatherIndices = gatherIndices.zip(data.shape.toList()) { index, shape ->
                if (index < 0) shape + index else index
            }.toIntArray()
            data[positiveGatherIndices]
        }

        val newStructure = BufferNDStructure(newStrides, newBuffer)
        return Tensor(null, newStructure, info.type)
    }

    fun reshape(shape: IntArray): Tensor {
        val newStrides = TensorStrides(shape)
        require(data.strides.linearSize == newStrides.linearSize) { "New shape is not compatible with the previous one" }

        val newBuffer = BufferNDStructure(newStrides, data.buffer)
        return Tensor(info.name, newBuffer, info.type)
    }

    fun reshape(tensorShape: Tensor): Tensor {
        val requestedShape = tensorShape.data.buffer as Buffer<Long>
        val requestedShapeArray = IntArray(requestedShape.size) { i -> requestedShape[i].toInt() }
        require(requestedShapeArray.count { it == -1 } <= 1) { "At most one dimension of the new shape can be -1" }

        val newShape = requestedShapeArray.copyOf()
        for ((i, axisShape) in requestedShapeArray.withIndex()) {
            if (axisShape == 0) newShape[i] = data.shape[i]
        }

        val negativeIdx = newShape.indexOf(-1)
        if (negativeIdx != -1) {
            val elementsCount = newShape.filter { it != -1 }.reduce(Int::times)
            newShape[negativeIdx] = data.strides.linearSize / elementsCount
        }

        return reshape(newShape)
    }

    fun squeeze(vararg axes: Int): Tensor {
        val actualAxes = if (axes.isNotEmpty()) {
            axes.map { indexAxis(it) }
        } else {
            data.shape.withIndex().filter { it.value == 1 }.map { it.index }
        }
        require(actualAxes.all { data.shape[it] == 1 })

        val shapeIndices = data.shape.indices - actualAxes
        val newShape = data.shape.sliceArray(shapeIndices)

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
            DataType.DOUBLE -> Tensor(proto.dims, proto.double_data.toDoubleArray().asBuffer(), type, proto.name)
            DataType.FLOAT -> Tensor(proto.dims, proto.float_data.toFloatArray().asBuffer(), type, proto.name)
            DataType.INT64 -> Tensor(proto.dims, proto.int64_data.toLongArray().asBuffer(), type, proto.name)
            DataType.INT32 -> Tensor(proto.dims, proto.int32_data.toIntArray().asBuffer(), type, proto.name)
            DataType.STRING -> Tensor(proto.dims, proto.string_data.map { it.utf8() }, type, proto.name)
            else -> error("Unsupported data type")
        }

        private operator fun invoke(name: String?, matrix: Matrix<*>, type: DataType): Tensor {
            if (matrix is BufferMatrix) {
                return Tensor(name, BufferNDStructure(TensorStrides(matrix.shape), matrix.buffer as Buffer<Any>), type)
            }
            val elements = matrix.elements().toList()
            val buffer = createBuffer(type, elements.size) { i -> elements[i].second }
            return Tensor(name, BufferNDStructure(TensorStrides(matrix.shape), buffer as Buffer<Any>), type)
        }

        operator fun invoke(dims: List<Long>, value: List<*>, type: DataType, name: String?): Tensor {
            val buffer = createBuffer(type, value.size) { i -> value[i]!! }
            val data = BufferNDStructure(TensorStrides(dims.toIntArray()), buffer)
            return Tensor(name, data, type)
        }

        operator fun invoke(dims: List<Long>, value: Buffer<*>, type: DataType, name: String?): Tensor {
            val data = BufferNDStructure(TensorStrides(dims.toIntArray()), value as Buffer<Any>)
            return Tensor(name, data, type)
        }

        operator fun invoke(value: List<Any>, type: DataType): Tensor {
            val dims = intArrayOf(value.size)
            val buffer = createBuffer(type, value.size) { i -> value[i] }
            val data = BufferNDStructure(TensorStrides(dims), buffer)
            return Tensor("out", data, type)
        }
    }
}

