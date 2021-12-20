package io.kinference.webgpu.utils

import io.kinference.TestLoggerFactory
import io.kinference.data.ONNXDataType
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.ArrayAssertions.assertArrayEquals
import io.kinference.webgpu.engine.WebGPUData
import io.kinference.webgpu.tensor.WebGPUTensor
import kotlin.math.abs
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

object WebGPUAssertions {
    @OptIn(ExperimentalUnsignedTypes::class)
    fun assertEquals(expected: WebGPUTensor, actual: WebGPUTensor, delta: Double) {
        assertEquals(expected.data.info.type, actual.data.info.type, "Types of tensors ${expected.name} do not match")
        assertContentEquals(expected.data.info.shape, actual.data.info.shape, "Shapes are incorrect")

        val typeInfo = expected.info
        when (typeInfo.type) {
            TensorProto.DataType.FLOAT -> {
                val expectedArray = expected.data.getMappedRange().toFloatArray()
                val actualArray = actual.data.getMappedRange().toFloatArray()

                assertArrayEquals(
                    expectedArray,
                    actualArray,
                    { l, r -> abs(l - r).toDouble() },
                    delta,
                    "Tensor ${expected.name} does not match"
                )
            }
            TensorProto.DataType.INT32 -> {
                val expectedArray = expected.data.getMappedRange().toIntArray()
                val actualArray = expected.data.getMappedRange().toIntArray()

                assertArrayEquals(
                    expectedArray,
                    actualArray,
                    { l, r -> abs(l - r).toDouble() },
                    delta,
                    "Tensor ${expected.name} does not match"
                )
            }
            TensorProto.DataType.UINT32 -> {
                val expectedArray = expected.data.getMappedRange().toUIntArray()
                val actualArray = expected.data.getMappedRange().toUIntArray()

                assertArrayEquals(
                    expectedArray,
                    actualArray,
                    { l, r -> abs((l - r).toDouble()) },
                    delta,
                    "Tensor ${expected.name} does not match"
                )
            }
            else -> assertEquals(expected, actual, "Tensor ${expected.name} does not match")
        }
    }

    fun assertEquals(expected: WebGPUData<*>, actual: WebGPUData<*>, delta: Double) {
        when (expected.type) {
            ONNXDataType.ONNX_TENSOR -> assertEquals(expected as WebGPUTensor, actual as WebGPUTensor, delta)
            ONNXDataType.ONNX_SEQUENCE -> TODO()
            ONNXDataType.ONNX_MAP -> TODO()
        }
    }
}
