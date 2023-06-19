package io.kinference.core.operators.convolution

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.DoubleNDArray
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.extensions.conv.conv
import io.kinference.ndarray.extensions.utils.InputInfo
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.toIntArray

sealed class Conv(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)  // last version. Other versions: 1.

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ConvVer11.VERSION.asRange() -> ConvVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Conv operator: $version")
            }
    }
}

class ConvVer11(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Conv(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(TensorProto.DataType.FLOAT16, TensorProto.DataType.FLOAT, TensorProto.DataType.DOUBLE)

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("auto_pad", setOf(AttributeProto.AttributeType.STRING), false, "NOTSET"),
            AttributeInfo("dilations", setOf(AttributeProto.AttributeType.INTS), false),
            AttributeInfo("group", setOf(AttributeProto.AttributeType.INT), false, 1),
            AttributeInfo("kernel_shape", setOf(AttributeProto.AttributeType.INTS), false),
            AttributeInfo("pads", setOf(AttributeProto.AttributeType.INTS), false),
            AttributeInfo("strides", setOf(AttributeProto.AttributeType.INTS), false)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false, differentiable = true),  // [batch_size, num_of_channels, D1, D2, ..., Dn]
            IOInfo(1, TYPE_CONSTRAINTS, "W", optional = false, differentiable = true),  // [num_of_feature_maps, num_of_channels `div` groups, k1, k2, ..., kn]
            IOInfo(2, TYPE_CONSTRAINTS, "B", optional = true, differentiable = true)    // [num_of_feature_maps]
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("Conv", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val autoPad: String by attribute("auto_pad")
    private val dilations: IntArray? by attributeOrNull { it: LongArray? -> it?.toIntArray() }
    private val group: Int by attribute { it: Number -> it.toInt() }
    private val kernelShape: IntArray? by attributeOrNull("kernel_shape") { it: LongArray? -> it?.toIntArray() }
    private val pads: IntArray? by attributeOrNull { it: LongArray? -> it?.toIntArray() }
    private val strides: IntArray? by attributeOrNull { it: LongArray? -> it?.toIntArray() }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val x = inputs[0]!!.data
        val w = inputs[1]!!.data
        val b = inputs.getOrNull(2)?.data

        val inputInfo = InputInfo.Builder()
            .specifyDimensions(x.rank - 2)
            .specifyInputShape(x.shape.sliceArray(2 .. x.shape.lastIndex))
            .specifyKernelShape(w.shape.sliceArray(2 .. w.shape.lastIndex))
            .specifyStrides(strides)
            .specifyGroups(group)
            .specifyDilations(dilations)
            .specifyPads(pads)
            .specifyAutoPad(autoPad)
            .build()

        val y = when (x.type) {
            DataType.FLOAT -> (x as FloatNDArray).conv(w as FloatNDArray, b as FloatNDArray?, inputInfo)
            DataType.DOUBLE -> (x as DoubleNDArray).conv(w as DoubleNDArray, b as DoubleNDArray?, inputInfo)
            else -> error("Data type ${x.type} is not supported.")
        }

        return listOf(y.asTensor("Y"))
    }
}
