package io.kinference.multik

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import io.kinference.ndarray.toLongArray
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.utils.ArrayAssertions
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.junit.jupiter.api.Test
import java.nio.IntBuffer
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ORTMultikAdapterTest {
    @Test
    fun test_multik_adapter_convert_to_onnx_data() {
        val array = IntArray(4) { it }
        val shape = intArrayOf(1, 2, 2)
        val multikArray = NDArray<Int, D3>(MemoryViewIntArray(array), shape = shape, dtype = DataType.IntDataType, dim = D3)
        val convertedTensor = ORTMultikTensorAdapter.toONNXData(ORTMultikData.MultikTensor("test", multikArray as MultiArray<Number, Dimension>))
        val expectedTensorData = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), IntBuffer.wrap(array), shape.toLongArray())
        val expectedTensor = ORTTensor("test", expectedTensorData)
        assertTensorEquals(expectedTensor, convertedTensor)
    }

    @Test
    fun test_multik_adapter_convert_from_onnx_data() {
        val array = IntArray(6) { it }
        val shape = intArrayOf(2, 3)
        val tensorData = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), IntBuffer.wrap(array), shape.toLongArray())
        val tensor = ORTTensor("test", tensorData)
        val expectedArray = NDArray<Int, D2>(MemoryViewIntArray(array), shape = shape, dtype = DataType.IntDataType, dim = D2)
        val convertedArray = ORTMultikTensorAdapter.fromONNXData(tensor).data
        assertTrue(expectedArray == convertedArray)
    }

    companion object {
        fun assertTensorEquals(expected: ORTTensor, actual: ORTTensor) {
            assertEquals(expected.data.info.type, actual.data.info.type, "Types of tensors ${expected.name} do not match")
            assertEquals(expected.name, actual.name, "Names of tensors do not match")
            ArrayAssertions.assertArrayEquals(expected.data.info.shape.toTypedArray(), actual.data.info.shape.toTypedArray(), "Shapes do not match")
            ArrayAssertions.assertArrayEquals(
                expected.data.intBuffer.array(),
                actual.data.intBuffer.array(),
                { l, r -> abs(l - r).toDouble() },
                delta = 0.0,
                ""
            )
        }
    }
}
