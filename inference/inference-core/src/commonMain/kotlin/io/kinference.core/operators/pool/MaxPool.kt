package io.kinference.core.operators.pool

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.operators.utils.*
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.toIntArray

sealed class MaxPool(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 12)  // last version. Other versions: 11, 10, 8, 1.

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in MaxPool12.VERSION.asRange() -> MaxPool12(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Greater operator: $version")
            }
    }
}

class MaxPool12(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    MaxPool(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS_T = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.INT8,
            TensorProto.DataType.UINT8
        )

        private val TYPE_CONSTRAINTS_I = setOf(
            TensorProto.DataType.INT64
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("auto_pad", setOf(AttributeProto.AttributeType.STRING), false, "NOTSET"),
            AttributeInfo("ceil_mode", setOf(AttributeProto.AttributeType.INT), false, 0),
            AttributeInfo("dilations", setOf(AttributeProto.AttributeType.INTS), false),
            AttributeInfo("kernel_shape", setOf(AttributeProto.AttributeType.INTS), true),
            AttributeInfo("pads", setOf(AttributeProto.AttributeType.INTS), false),
            AttributeInfo("storage_order", setOf(AttributeProto.AttributeType.INT), false, 0),
            AttributeInfo("strides", setOf(AttributeProto.AttributeType.INTS), false)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS_T, "X", optional = false, differentiable = true),  // [batch_size, num_of_channels, D1, D2, ..., Dn]
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS_T, "Y", optional = false, differentiable = true),
            IOInfo(1, TYPE_CONSTRAINTS_I, "Indices", optional = true, differentiable = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 12)
        private val INFO = OperatorInfo("MaxPool", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val autoPad: String by attribute("auto_pad")
    private val ceilMode: Int by attribute("ceil_mode") { it: Number -> it.toInt() }
    private val dilations: IntArray? by attributeOrNull("dilations") { it: LongArray? -> it?.toIntArray() }
    private val kernelShape: IntArray by attribute("kernel_shape") { it: LongArray -> it.toIntArray() }
    private val pads: IntArray? by attributeOrNull("pads") { it: LongArray? -> it?.toIntArray() }
    private val storageOrder: Int by attribute("storage_order") { it: Number -> it.toInt() }
    private val strides: IntArray? by attributeOrNull("strides") { it: LongArray? -> it?.toIntArray() }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val x = inputs[0]!!.data

        val parsedStrides = parseStrides(strides, x.shape.size)
        val parsedPads = parsePads(autoPad, pads, x.shape, IntArray(kernelShape.size + 2) { i -> if (i < 2) 1 else kernelShape[i - 2] }, parsedStrides)
        val parsedDilations = parseDilations(dilations, x.shape.size)
        val computeIndices = if (outputs.size == 1) -1 else storageOrder

        val results = when (x.type) {
            DataType.FLOAT -> (x as FloatNDArray).maxPool(kernelShape, parsedPads, parsedStrides, parsedDilations, ceilMode, computeIndices, -Float.MAX_VALUE)
            DataType.DOUBLE -> (x as DoubleNDArray).maxPool(kernelShape, parsedPads, parsedStrides, parsedDilations, ceilMode, computeIndices, -Double.MAX_VALUE)
            DataType.UBYTE -> (x as UByteNDArray).maxPool(kernelShape, parsedPads, parsedStrides, parsedDilations, ceilMode, computeIndices)
            DataType.BYTE -> (x as ByteNDArray).maxPool(kernelShape, parsedPads, parsedStrides, parsedDilations, ceilMode, computeIndices)
            else -> {
                throw IllegalArgumentException("Data type ${x.type} is not supported.")
            }
        }

        if (outputs.size == 1)
            return listOf(results[0].asTensor("Y"))

        return listOf(results[0].asTensor("Y"), results[1].asTensor("Indices"))
    }
}
