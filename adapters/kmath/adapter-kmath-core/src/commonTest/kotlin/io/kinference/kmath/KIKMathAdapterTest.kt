package io.kinference.kmath

import io.kinference.core.KIONNXData
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.model.KIModel
import io.kinference.data.ONNXDataType
import io.kinference.kmath.KIKMathData.*
import io.kinference.ndarray.arrays.IntNDArray
import io.kinference.ndarray.extensions.createNDArray
import io.kinference.ndarray.extensions.tiledFromPrimitiveArray
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.*
import io.kinference.types.TensorShape
import io.kinference.types.ValueTypeInfo
import io.kinference.utils.ArrayAssertions
import io.kinference.utils.TestRunner
import space.kscience.kmath.nd.*
import space.kscience.kmath.structures.Buffer
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals

class KIKMathAdapterTest {
    @Test
    fun test_kmath_adapter_convert_to_onnx_tensor() {
        val array = IntArray(4) { it }
        val shape = intArrayOf(1, 2, 2)
        val kmathArray = BufferND(DefaultStrides(shape), Buffer.auto(shape.reduce(Int::times)) { array[it] })
        val convertedTensor = KIKMathTensorAdapter.toONNXData(KMathTensor("test", kmathArray))
        val expectedTensor = createNDArray(DataType.INT, tiledFromPrimitiveArray(shape, array), shape).asTensor("test")
        assertKIEquals(expectedTensor, convertedTensor)
    }

    @Test
    fun test_kmath_adapter_convert_to_onnx_map() {
        val array = IntArray(4) { it }
        val shape = intArrayOf(1, 2, 2)

        val kmathArray = KMathTensor("", BufferND(DefaultStrides(shape), Buffer.auto(shape.reduce(Int::times)) { array[it] }))
        val kmathSequence = KMathSequence("", listOf(kmathArray))
        val kmathMap = mapOf(0 to kmathSequence, 1 to kmathSequence, 2 to kmathSequence)
        val convertedMap = KIKMathMapAdapter.toONNXData(KMathMap("test", kmathMap as Map<Any, KIKMathData<*>>))

        val tensorInfo = ValueTypeInfo.TensorTypeInfo(TensorShape(shape), TensorProto.DataType.INT32)
        val tensor = KITensor(null, IntNDArray(shape) { it }, tensorInfo)
        val expectedValueInfo = ValueTypeInfo.MapTypeInfo(TensorProto.DataType.INT32, ValueTypeInfo.SequenceTypeInfo(tensorInfo))
        val expectedMapData = mapOf(
            0 to KIONNXSequence(null, listOf(tensor), ValueTypeInfo.SequenceTypeInfo(tensorInfo)),
            1 to KIONNXSequence(null, listOf(tensor), ValueTypeInfo.SequenceTypeInfo(tensorInfo)),
            2 to KIONNXSequence(null, listOf(tensor), ValueTypeInfo.SequenceTypeInfo(tensorInfo))
        ) as Map<Any, KIONNXData<*>>
        val expectedMap = KIONNXMap("test", expectedMapData, expectedValueInfo)
        assertKIEquals(expectedMap, convertedMap)
    }

    @Test
    fun test_kmath_adapter_convert_to_onnx_sequence() {
        val array = IntArray(4) { it }
        val shape = intArrayOf(1, 2, 2)
        val kmathArray = KMathTensor("", BufferND(DefaultStrides(shape), Buffer.auto(shape.reduce(Int::times)) { array[it] }))
        val kmathSeq = KMathSequence("", List(4) { KMathSequence("", listOf(kmathArray)) })
        val convertedSeq = KIKMathSequenceAdapter.toONNXData(kmathSeq)

        val tensorInfo = ValueTypeInfo.TensorTypeInfo(TensorShape(shape), TensorProto.DataType.INT32)
        val expectedValueInfo = ValueTypeInfo.SequenceTypeInfo(ValueTypeInfo.SequenceTypeInfo(tensorInfo))
        val expectedSeqData = KIONNXSequence("", List(1) { KITensor("", IntNDArray(shape) { it }, tensorInfo) }, ValueTypeInfo.SequenceTypeInfo(tensorInfo))
        val expectedSeq = KIONNXSequence("test", List(4) { expectedSeqData }, expectedValueInfo)
        assertKIEquals(expectedSeq, convertedSeq)
    }

    @Test
    fun test_kmath_adapter_convert_from_onnx_tensor() {
        val array = IntArray(6) { it }
        val shape = intArrayOf(2, 3)
        val tensor = createNDArray(DataType.INT, tiledFromPrimitiveArray(shape, array), shape).asTensor()
        val expectedArray = KMathTensor("", BufferND(DefaultStrides(shape), Buffer.auto(shape.reduce(Int::times)) { array[it] }))
        val convertedArray = KIKMathTensorAdapter.fromONNXData(tensor)
        StructureND.contentEquals(expectedArray.data as StructureND<Int>, convertedArray.data as StructureND<Int>)
    }

    @Test
    fun test_kmath_adapter_inference() = TestRunner.runTest {
        val inOutTensorType = TypeProto.Tensor(TensorProto.DataType.FLOAT, TensorShapeProto(listOf(TensorShapeProto.Dimension(dimValue = 6L))))
        val modelProto = ModelProto(
            graph = GraphProto(
                input = mutableListOf(ValueInfoProto("input", TypeProto(tensorType = inOutTensorType))),
                output = mutableListOf(ValueInfoProto("output", TypeProto(tensorType = inOutTensorType))),
                node = mutableListOf(NodeProto(input = mutableListOf("input"), mutableListOf("output"), opType = "Identity"))
            )
        )
        val modelAdapter = KIKMathModelAdapter(KIModel(modelProto))
        val array = FloatArray(6) { it.toFloat() }
        val shape = intArrayOf(6)
        val inputArray = BufferND(DefaultStrides(shape), Buffer.auto(shape.reduce(Int::times)) { array[it] })
        val result = modelAdapter.predict(listOf(KMathTensor("input", inputArray)))
        StructureND.contentEquals(inputArray, result.values.first().data as StructureND<Float>)
    }

    companion object {
        fun assertEquals(expected: KITensor, actual: KITensor) {
            assertEquals(expected.data.type, actual.data.type, "Types of tensors ${expected.name} do not match")
            ArrayAssertions.assertArrayEquals(expected.data.shape.toTypedArray(), actual.data.shape.toTypedArray(), "Shapes do not match")
            ArrayAssertions.assertArrayEquals(
                (expected.data as IntNDArray).array,
                (actual.data as IntNDArray).array,
                { l, r -> abs(l - r).toDouble() },
                delta = 0.0,
                ""
            )
        }

        fun assertEquals(expected: KIONNXMap, actual: KIONNXMap) {
            assertEquals(expected.keyType, actual.keyType, "Map key types should match")
            assertEquals(expected.data.keys, actual.data.keys, "Map key sets are not equal")

            for (entry in expected.data.entries) {
                assertKIEquals(entry.value, actual.data[entry.key]!!)
            }
        }

        fun assertEquals(expected: KIONNXSequence, actual: KIONNXSequence) {
            assertEquals(expected.length, actual.length, "Sequence lengths do not match")

            for (i in expected.data.indices) {
                assertKIEquals(expected.data[i], actual.data[i])
            }
        }

        fun assertKIEquals(expected: KIONNXData<*>, actual: KIONNXData<*>) {
            when (expected.type) {
                ONNXDataType.ONNX_TENSOR -> assertEquals(expected as KITensor, actual as KITensor)
                ONNXDataType.ONNX_MAP -> assertEquals(expected as KIONNXMap, actual as KIONNXMap)
                ONNXDataType.ONNX_SEQUENCE -> assertEquals(expected as KIONNXSequence, actual as KIONNXSequence)
            }
        }
    }
}
