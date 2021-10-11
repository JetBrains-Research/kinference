package io.kinference.adapters

import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.kmath.KMathStructureNDAdapter
import io.kinference.ndarray.arrays.IntNDArray
import io.kinference.ndarray.extensions.createArray
import io.kinference.ndarray.extensions.createNDArray
import io.kinference.primitives.types.DataType
import io.kinference.utils.ArrayAssertions
import space.kscience.kmath.nd.*
import space.kscience.kmath.structures.Buffer
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals

class KMathStructureNDAdapterTest {
    @Test
    fun test_kmath_adapter_convert_to_onnx_data() {
        val array = IntArray(4) { it }
        val shape = intArrayOf(1, 2, 2)
        val kmathArray = BufferND(DefaultStrides(shape), Buffer.auto(shape.reduce(Int::times)) { array[it] })
        val convertedTensor = KMathStructureNDAdapter.toONNXData("test", kmathArray)
        val expectedTensor = createNDArray(DataType.INT, createArray(shape, array), shape).asTensor("test")
        assertTensorEquals(expectedTensor, convertedTensor)
    }

    @Test
    fun test_kmath_adapter_convert_from_onnx_data() {
        val array = IntArray(6) { it }
        val shape = intArrayOf(2, 3)
        val tensor = createNDArray(DataType.INT, createArray(shape, array), shape).asTensor()
        val expectedArray = BufferND(DefaultStrides(shape), Buffer.auto(shape.reduce(Int::times)) { array[it] })
        val convertedArray = KMathStructureNDAdapter.fromONNXData(tensor) as StructureND<Int>
        StructureND.contentEquals(expectedArray, convertedArray)
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
