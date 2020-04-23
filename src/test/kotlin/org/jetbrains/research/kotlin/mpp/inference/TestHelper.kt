package org.jetbrains.research.kotlin.mpp.inference

import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.space.SpaceStrides
import org.jetbrains.research.kotlin.mpp.inference.space.resolveSpace
import org.jetbrains.research.kotlin.mpp.inference.space.toIntArray
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import org.junit.jupiter.api.Assertions.*
import scientifik.kmath.structures.*
import java.io.File
import java.nio.ByteBuffer
import kotlin.math.pow


fun getTensor(path: File) : Tensor<*> {
    val tensorProto = TensorProto.ADAPTER.decode(path.readBytes())
    return when (DataType.fromValue(tensorProto.data_type!!) ?: 0) {
        DataType.FLOAT -> getTensorFloat(tensorProto)
        else -> throw UnsupportedOperationException()
    }
}

fun getTensorFloat(tensorProto : TensorProto) : Tensor<Float> {
    val data = tensorProto.raw_data!!.toByteArray().asIterable().chunked(4) { ByteBuffer.wrap(it.reversed().toByteArray()).float }.asBuffer()
    val structure = BufferNDStructure(SpaceStrides(tensorProto.dims.toIntArray()), data)
    return Tensor(tensorProto.name, structure, DataType.FLOAT, resolveSpace(tensorProto.dims))
}

val delta = (10.0).pow(-7)

fun assertTensors(expected: Tensor<*>, actual: Tensor<*>) {
    assertEquals(expected.type, actual.type)
    @Suppress("UNCHECKED_CAST")
    when (expected.type) {
        DataType.FLOAT -> {
            expected as Tensor<Float>
            actual as Tensor<Float>
            expected.data.buffer.asIterable().forEachIndexed() { index, value ->
                assertEquals(value, actual.data.buffer[index], delta.toFloat())
            }
        }

        DataType.DOUBLE -> {
            expected as Tensor<Double>
            actual as Tensor<Double>
            expected.data.buffer.asIterable().forEachIndexed() { index, value ->
                assertEquals(value, actual.data.buffer[index], delta)
            }
        }

        else -> assertEquals(expected, actual)
    }
}
