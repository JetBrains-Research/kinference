package io.kinference.kmath

import ai.onnxruntime.*
import io.kinference.ndarray.toLongArray
import io.kinference.ort_gpu.data.tensor.ORTGPUTensor
import io.kinference.utils.ArrayAssertions
import space.kscience.kmath.nd.*
import space.kscience.kmath.structures.Buffer
import java.nio.ByteBuffer
import java.nio.IntBuffer
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals

class ORTGPUKMathAdapterTest {
    @Test
    fun test_kmath_adapter_convert_to_onnx_data() {
        val array = IntArray(4) { it }
        val shape = intArrayOf(1, 2, 2)
        val kmathArray = BufferND(DefaultStrides(shape), Buffer.auto(shape.reduce(Int::times)) { array[it] })
        val convertedTensor = ORTGPUKMathTensorAdapter.toONNXData(ORTGPUKMathData.KMathTensor("test", kmathArray))
        val expectedTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), IntBuffer.wrap(array), shape.toLongArray())
        val expectedOrtTensor = ORTGPUTensor("test", expectedTensor)
        assertTensorEquals(expectedOrtTensor, convertedTensor)
    }

    @Test
    fun test_kmath_adapter_convert_to_onnx_data_ubyte() {
        val array = UByteArray(4) { it.toUByte() }
        val shape = intArrayOf(1, 2, 2)
        val kmathArray = BufferND(DefaultStrides(shape), Buffer.auto(shape.reduce(Int::times)) { array[it] })
        val convertedTensor = ORTGPUKMathTensorAdapter.toONNXData(ORTGPUKMathData.KMathTensor("test", kmathArray))
        val expectedTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), ByteBuffer.wrap(array.toByteArray()), shape.toLongArray(), OnnxJavaType.UINT8)
        val expectedOrtTensor = ORTGPUTensor("test", expectedTensor)
        assertTensorEquals(expectedOrtTensor, convertedTensor)
    }

    @Test
    fun test_kmath_adapter_convert_from_onnx_data() {
        val array = IntArray(6) { it }
        val shape = intArrayOf(2, 3)
        val tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), IntBuffer.wrap(array), shape.toLongArray())
        val ortTensor = ORTGPUTensor("test", tensor)
        val expectedArray = BufferND(DefaultStrides(shape), Buffer.auto(shape.reduce(Int::times)) { array[it] })
        val convertedArray = ORTGPUKMathTensorAdapter.fromONNXData(ortTensor).data as StructureND<Int>
        StructureND.contentEquals(expectedArray, convertedArray)
    }

    companion object {
        fun assertTensorEquals(expected: ORTGPUTensor, actual: ORTGPUTensor) {
            assertEquals(expected.data.info.type, actual.data.info.type, "Types of tensors ${expected.name} do not match")
            assertEquals(expected.name, actual.name, "Names of tensors do not match")
            ArrayAssertions.assertArrayEquals(expected.data.info.shape.toTypedArray(), actual.data.info.shape.toTypedArray(), "Shapes do not match")
            when {
                expected.data.intBuffer != null -> assertIntEquals(expected, actual)
                expected.data.byteBuffer != null -> assertByteEquals(expected, actual)
            }
        }

        fun assertIntEquals(expected: ORTGPUTensor, actual: ORTGPUTensor) = ArrayAssertions.assertArrayEquals(
            expected.data.intBuffer.array(),
            actual.data.intBuffer.array(),
            { l, r -> abs(l - r).toDouble() },
            delta = 0.0,
            ""
        )

        fun assertByteEquals(expected: ORTGPUTensor, actual: ORTGPUTensor) = ArrayAssertions.assertArrayEquals(
            expected.data.byteBuffer.array(),
            actual.data.byteBuffer.array(),
            { l, r -> abs(l - r).toDouble() },
            delta = 0.0,
            ""
        )
    }
}