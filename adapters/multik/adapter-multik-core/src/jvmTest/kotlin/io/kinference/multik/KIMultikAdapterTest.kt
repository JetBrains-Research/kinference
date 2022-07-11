package io.kinference.multik

import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.ndarray.arrays.IntNDArray
import io.kinference.ndarray.extensions.createArray
import io.kinference.ndarray.extensions.createNDArray
import io.kinference.primitives.types.DataType
import io.kinference.utils.ArrayAssertions
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.junit.jupiter.api.Test
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class KIMultikAdapterTest {
    @Test
    fun test_multik_adapter_convert_to_onnx_data() {
        val array = IntArray(4) { it }
        val shape = intArrayOf(1, 2, 2)
        val multikArray = NDArray<Int, D3>(MemoryViewIntArray(array), shape = shape/*, dtype = MultikDataType.IntDataType*/, dim = D3)
        val convertedTensor = KIMultikTensorAdapter.toONNXData(KIMultikData.MultikTensor("test", multikArray as MultiArray<Number, Dimension>))
        val expectedTensor = createNDArray(DataType.INT, createArray(shape, array), shape).asTensor("test")
        assertTensorEquals(expectedTensor, convertedTensor)
    }

    @Test
    fun test_multik_adapter_convert_from_onnx_data() {
        val array = IntArray(6) { it }
        val shape = intArrayOf(2, 3)
        val tensor = createNDArray(DataType.INT, createArray(shape, array), shape).asTensor()
        val expectedArray = NDArray<Int, D2>(MemoryViewIntArray(array), shape = shape/*, dtype = MultikDataType.IntDataType*/, dim = D2)
        val convertedArray = KIMultikTensorAdapter.fromONNXData(tensor).data
        assertTrue(expectedArray == convertedArray)
    }

    companion object {
        fun assertTensorEquals(expected: KITensor, actual: KITensor) {
            assertEquals(expected.data.type, actual.data.type, "Types of tensors ${expected.name} do not match")
            assertEquals(expected.name, actual.name, "Names of tensors do not match")
            ArrayAssertions.assertArrayEquals(expected.data.shape.toTypedArray(), actual.data.shape.toTypedArray(), "Shapes do not match")
            ArrayAssertions.assertArrayEquals(
                (expected.data as IntNDArray).array,
                (actual.data as IntNDArray).array,
                { l, r -> abs(l - r).toDouble() },
                delta = 0.0,
                ""
            )
        }
    }
}
