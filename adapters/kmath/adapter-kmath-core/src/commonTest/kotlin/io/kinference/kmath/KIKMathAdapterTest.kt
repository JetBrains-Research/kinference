package io.kinference.kmath

import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.model.KIModel
import io.kinference.ndarray.arrays.IntNDArray
import io.kinference.ndarray.extensions.createArray
import io.kinference.ndarray.extensions.createNDArray
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.ModelProto
import io.kinference.utils.ArrayAssertions
import space.kscience.kmath.nd.*
import space.kscience.kmath.structures.Buffer
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.time.ExperimentalTime

class KIKMathAdapterTest {
    private val emptyModel = KIModel(ModelProto(graph = GraphProto()))
    private val testAdapter = KIKMathAdapter(emptyModel)

    @OptIn(ExperimentalTime::class)
    @Test
    fun test_kmath_adapter_convert_to_onnx_data() {
        val array = IntArray(4) { it }
        val shape = intArrayOf(1, 2, 2)
        val kmathArray = NDBuffer(DefaultStrides(shape), Buffer.auto(shape.reduce(Int::times)) { array[it] })
        val convertedTensor = testAdapter.toONNXData("test", kmathArray)
        val expectedTensor = createNDArray(DataType.INT, createArray(shape, array), shape).asTensor("test")
        assertTensorEquals(expectedTensor, convertedTensor)
    }

    @OptIn(ExperimentalTime::class)
    @Test
    fun test_kmath_adapter_convert_from_onnx_data() {
        val array = IntArray(6) { it }
        val shape = intArrayOf(2, 3)
        val tensor = createNDArray(DataType.INT, createArray(shape, array), shape).asTensor()
        val expectedArray = NDBuffer(DefaultStrides(shape), Buffer.auto(shape.reduce(Int::times)) { array[it] })
        val convertedArray = testAdapter.fromONNXData(tensor) as NDStructure<Int>
        NDStructure.contentEquals(expectedArray, convertedArray)
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
