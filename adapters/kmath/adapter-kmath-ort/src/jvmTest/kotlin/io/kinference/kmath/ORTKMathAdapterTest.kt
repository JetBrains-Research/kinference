package io.kinference.kmath

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import io.kinference.ndarray.toLongArray
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.utils.ArrayAssertions
import space.kscience.kmath.nd.*
import space.kscience.kmath.structures.Buffer
import java.nio.IntBuffer
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.time.ExperimentalTime

class ORTKMathAdapterTest {
    @OptIn(ExperimentalTime::class)
    @Test
    fun test_kmath_adapter_convert_to_onnx_data() {
        val array = IntArray(4) { it }
        val shape = intArrayOf(1, 2, 2)
        val kmathArray = NDBuffer(DefaultStrides(shape), Buffer.auto(shape.reduce(Int::times)) { array[it] })
        val convertedTensor = ORTKMathTensorAdapter.toONNXData(ORTKMathData.KMathTensor("test", kmathArray))
        val expectedTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), IntBuffer.wrap(array), shape.toLongArray())
        val expectedOrtTensor = ORTTensor("test", expectedTensor)
        assertTensorEquals(expectedOrtTensor, convertedTensor)
    }

    @OptIn(ExperimentalTime::class)
    @Test
    fun test_kmath_adapter_convert_from_onnx_data() {
        val array = IntArray(6) { it }
        val shape = intArrayOf(2, 3)
        val tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), IntBuffer.wrap(array), shape.toLongArray())
        val ortTensor = ORTTensor("test", tensor)
        val expectedArray = NDBuffer(DefaultStrides(shape), Buffer.auto(shape.reduce(Int::times)) { array[it] })
        val convertedArray = ORTKMathTensorAdapter.fromONNXData(ortTensor).data as NDStructure<Int>
        NDStructure.contentEquals(expectedArray, convertedArray)
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
