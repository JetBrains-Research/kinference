package io.kinference.core.operators.layer.normalization

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.mapTo
import io.kinference.ndarray.arrays.tiled.DoubleTiledArray
import io.kinference.ndarray.arrays.tiled.FloatTiledArray
import io.kinference.ndarray.extensions.batchNorm.batchNorm
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto

sealed class BatchNormalization(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 9)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): BatchNormalization {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in BatchNormalizationVer9.VERSION.asRange() -> BatchNormalizationVer9(name, attributes, inputs, outputs)
                else -> error("Unsupported version of BatchNormalization operator: $version")
            }
        }
    }
}

class BatchNormalizationVer9 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : BatchNormalization(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "scale", optional = false),
            IOInfo(2, TYPE_CONSTRAINTS, "B", optional = false),
            IOInfo(3, TYPE_CONSTRAINTS, "input_mean", optional = false),
            IOInfo(4, TYPE_CONSTRAINTS, "input_var", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("epsilon", setOf(AttributeProto.AttributeType.FLOAT), required = false, default = 1e-05),
            AttributeInfo("momentum", setOf(AttributeProto.AttributeType.FLOAT), required = false, default = 0.9),
            AttributeInfo("training_mode", setOf(AttributeProto.AttributeType.INT), required = false, default = 0L),
        )

        internal val VERSION = VersionInfo(sinceVersion = 9)
        private val INFO = OperatorInfo("BatchNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.DEFAULT_DOMAIN)


        private suspend fun NumberNDArray.toFloatNDArray() = when (this) {
            is FloatNDArray -> this
            is DoubleNDArray-> {
                val result = FloatNDArray(FloatTiledArray(strides), strides)
                this.array.pointer().mapTo(result.array.pointer(), linearSize) { it.toFloat() }
                result
            }
            else -> error("Unsupported data type: $type")
        }

        private suspend fun NumberNDArray.toDoubleNDArray() = when (this) {
            is DoubleNDArray -> this
            is FloatNDArray-> {
                val result = DoubleNDArray(DoubleTiledArray(strides), strides)
                this.array.pointer().mapTo(result.array.pointer(), linearSize) { it.toDouble() }
                result
            }
            else -> error("Unsupported data type: $type")
        }
    }

    private val epsilon: Number by attribute()
    private val momentum: Number by attribute()
    private val trainingMode: Boolean by attribute("training_mode") { it: Number -> it.toInt() != 0 }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        require(!trainingMode) { "Training mode is not supported for BatchNormalization operator" }

        val input = inputs[0]!!.data as NumberNDArrayCore
        val scale = inputs[1]!!.data as NumberNDArrayCore
        val bias = inputs[2]!!.data as NumberNDArrayCore
        val mean = inputs[3]!!.data as NumberNDArrayCore
        val variance = inputs[4]!!.data as NumberNDArrayCore

        val output = when (input) {
            is FloatNDArray -> input.batchNorm(scale.toFloatNDArray(), bias.toFloatNDArray(), mean.toFloatNDArray(), variance.toFloatNDArray(), epsilon.toFloat())
            is DoubleNDArray -> input.batchNorm(scale.toDoubleNDArray(), bias.toDoubleNDArray(), mean.toDoubleNDArray(), variance.toDoubleNDArray(), epsilon.toDouble())
            else -> error("Unsupported data type: ${input.type}")
        }

        return listOf(output.asTensor("Y"))
    }
}
