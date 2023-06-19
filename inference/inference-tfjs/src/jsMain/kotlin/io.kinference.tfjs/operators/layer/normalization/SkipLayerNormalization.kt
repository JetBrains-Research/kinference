package io.kinference.tfjs.operators.layer.normalization

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class SkipLayerNormalization(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in SkipLayerNormalizationVer1.VERSION.asRange() -> SkipLayerNormalizationVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of SkipLayerNormalization operator: $version")
            }
    }
}

class SkipLayerNormalizationVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    SkipLayerNormalization(name, INFO, attributes, inputs, outputs) {

    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.FLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("epsilon", setOf(AttributeProto.AttributeType.FLOAT), false, 0.00001f)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "input", false),
            IOInfo(1, TYPE_CONSTRAINTS, "skip", false),
            IOInfo(2, TYPE_CONSTRAINTS, "gamma", false),
            IOInfo(3, TYPE_CONSTRAINTS, "beta", false),
            IOInfo(4, TYPE_CONSTRAINTS, "bias", true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "output", false),
            //Only for training, not supported now
            IOInfo(1, TYPE_CONSTRAINTS, "mean", true),
            IOInfo(2, TYPE_CONSTRAINTS, "inv_std_var", true)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("SkipLayerNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.ORT_DOMAIN)
    }

    private val epsilon: Float by attribute()

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data as NumberNDArrayTFJS
        val skip = inputs[1]!!.data as NumberNDArrayTFJS
        val gamma = inputs[2]!!.data as NumberNDArrayTFJS
        val beta = inputs[3]!!.data as NumberNDArrayTFJS
        val bias = inputs.getOrNull(4)?.data as? NumberNDArrayTFJS
        val out = tidyNDArray {
            val skippedInput = if (bias != null) {
                input.add(skip, bias.broadcastTo(input.shapeArray))
            } else {
                input.plus(skip)
            }

            val (mean, variance) = skippedInput.moments(axis = -1, keepDims = true)

            val epsilonTensor = NDArrayTFJS.floatScalar(epsilon)
            return@tidyNDArray (skippedInput - mean) / (variance + epsilonTensor).sqrt() * gamma + beta
        }
        return listOf(out.asTensor("output"))
    }
}
