package io.kinference.webgpu.utils

import io.kinference.data.ONNXDataType
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.ArrayAssertions.assertArrayEquals
import io.kinference.webgpu.engine.WebGPUData
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.ndarray.*
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
                val expectedArray = (expected.data.getData() as FloatNDArrayData).data
                val actualArray = (actual.data.getData() as FloatNDArrayData).data

                assertArrayEquals(
                    expectedArray,
                    actualArray,
                    { l, r -> abs(l - r).toDouble() },
                    delta,
                    "Tensor ${expected.name} does not match"
                )
            }
            TensorProto.DataType.INT32 -> {
                val expectedArray = (expected.data.getData() as IntNDArrayData).data
                val actualArray = (expected.data.getData() as IntNDArrayData).data

                assertArrayEquals(
                    expectedArray,
                    actualArray,
                    { l, r -> abs(l - r).toDouble() },
                    delta,
                    "Tensor ${expected.name} does not match"
                )
            }
            TensorProto.DataType.UINT32 -> {
                val expectedArray = (expected.data.getData() as UIntNDArrayData).data
                val actualArray = (expected.data.getData() as UIntNDArrayData).data

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
