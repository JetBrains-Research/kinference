package io.kinference.kmath

import ai.onnxruntime.*
import io.kinference.ort.data.tensor.ORTTensor
import io.kinference.utils.*
import space.kscience.kmath.nd.*
import space.kscience.kmath.structures.Buffer
import java.nio.ByteBuffer
import java.nio.IntBuffer
import kotlin.test.Test
import kotlin.test.assertEquals

class ORTKMathAdapterTest {
    @Test
    fun gpu_test_kmath_adapter_convert_to_onnx_data() = TestRunner.runTest {
        val array = IntArray(4) { it }
        val shape = intArrayOf(1, 2, 2)
        val kmathArray = BufferND(Strides(ShapeND(shape)), Buffer.auto(shape.reduce(Int::times)) { array[it] })
        val convertedTensor = ORTKMathTensorAdapter.toONNXData(ORTKMathData.KMathTensor("test", kmathArray))
        val expectedTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), IntBuffer.wrap(array), shape.toLongArray())
        val expectedOrtTensor = ORTTensor("test", expectedTensor)
        assertTensorEquals(expectedOrtTensor, convertedTensor)
    }

    @Test
    fun gpu_test_kmath_adapter_convert_to_onnx_data_ubyte() = TestRunner.runTest {
        val array = UByteArray(4) { it.toUByte() }
        val shape = intArrayOf(1, 2, 2)
        val kmathArray = BufferND(Strides(ShapeND(shape)), Buffer.auto(shape.reduce(Int::times)) { array[it] })
        val convertedTensor = ORTKMathTensorAdapter.toONNXData(ORTKMathData.KMathTensor("test", kmathArray))
        val expectedTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), ByteBuffer.wrap(array.toByteArray()), shape.toLongArray(), OnnxJavaType.UINT8)
        val expectedOrtTensor = ORTTensor("test", expectedTensor)
        assertTensorEquals(expectedOrtTensor, convertedTensor)
    }

    @Test
    fun gpu_test_kmath_adapter_convert_from_onnx_data() {
        val array = IntArray(6) { it }
        val shape = intArrayOf(2, 3)
        val tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), IntBuffer.wrap(array), shape.toLongArray())
        val ortTensor = ORTTensor("test", tensor)
        val expectedArray = BufferND(Strides(ShapeND(shape)), Buffer.auto(shape.reduce(Int::times)) { array[it] })
        val convertedArray = ORTKMathTensorAdapter.fromONNXData(ortTensor).data as StructureND<Int>
        StructureND.contentEquals(expectedArray, convertedArray)
    }

    companion object {
        fun assertTensorEquals(expected: ORTTensor, actual: ORTTensor) {
            assertEquals(expected.data.info.type, actual.data.info.type, "Types of tensors ${expected.name} do not match")
            assertEquals(expected.name, actual.name, "Names of tensors do not match")
            ArrayAssertions.assertArrayEquals(expected.data.info.shape.toTypedArray(), actual.data.info.shape.toTypedArray()) { "Shapes of tensors ${expected.name} do not match" }

            when (expected.data.info.type) {
                OnnxJavaType.INT32 -> ArrayAssertions.assertArrayEquals(expected.toIntArray(), actual.toIntArray(), delta = 0.0) { "Tensors ${expected.name} do not match" }
                OnnxJavaType.UINT8 -> ArrayAssertions.assertArrayEquals(expected.toUByteArray(), actual.toUByteArray(), delta = 0.0) { "Tensors ${expected.name} do not match"}
                else -> error("Unsupported data type: ${expected.data.info.type}")
            }
        }
    }
}
