package io.kinference.core.operators.ml.svm

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.operators.ml.utils.LabelsInfo
import io.kinference.core.operators.ml.utils.LabelsInfo.Companion.getLabelsInfo
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.FloatTiledArray
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto

sealed class SVMClassifier(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): SVMClassifier {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in SVMClassifierVer1.VERSION.asRange() -> SVMClassifierVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of SVMClassifier operator: $version")
            }
        }
    }
}

class SVMClassifierVer1 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : SVMClassifier(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.INT64, TensorProto.DataType.STRING), "Y", optional = false),
            IOInfo(1, setOf(TensorProto.DataType.FLOAT), "Z", optional = false),
        )

        private val labelsIntAttributeInfo = AttributeInfo("classlabels_ints", setOf(AttributeProto.AttributeType.INTS), required = false)
        private val labelsStringAttributeInfo = AttributeInfo("classlabels_strings", setOf(AttributeProto.AttributeType.STRINGS), required = false)

        private val ATTRIBUTES_INFO = listOf(
            labelsIntAttributeInfo,
            labelsStringAttributeInfo,
            AttributeInfo("coefficients", setOf(AttributeProto.AttributeType.FLOATS), required = true),
            AttributeInfo("kernel_params", setOf(AttributeProto.AttributeType.FLOATS), required = false, default = floatArrayOf()),
            AttributeInfo("kernel_type", setOf(AttributeProto.AttributeType.STRING), required = false, default = "LINEAR"),
            AttributeInfo("post_transform", setOf(AttributeProto.AttributeType.STRING), required = false, default = "NONE"),
            AttributeInfo("prob_a", setOf(AttributeProto.AttributeType.FLOATS), required = false, default = floatArrayOf()),
            AttributeInfo("prob_b", setOf(AttributeProto.AttributeType.FLOATS), required = false, default = floatArrayOf()),
            AttributeInfo("rho", setOf(AttributeProto.AttributeType.FLOATS), required = true),
            AttributeInfo("support_vectors", setOf(AttributeProto.AttributeType.FLOATS), required = false, default = floatArrayOf()),
            AttributeInfo("vectors_per_class", setOf(AttributeProto.AttributeType.INTS), required = false, default = longArrayOf()),
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("SVMClassifier", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.ML_DOMAIN)
    }

    private val labels = getLabelsInfo(
        intLabelsName = labelsIntAttributeInfo.name,
        stringLabelsName = labelsStringAttributeInfo.name
    )

    private val svmInfo = SvmInfo(
        getAttribute("coefficients"),
        getAttributeOrNull("kernel_params"),
        getAttributeOrNull("kernel_type"),
        getAttributeOrNull("post_transform"),
        getAttribute("prob_a"),
        getAttribute("prob_b"),
        getAttribute("rho"),
        getAttribute("support_vectors"),
        getAttribute("vectors_per_class"),
        labels.size
    )

    private val svm = SvmCommon.fromInfo(svmInfo, labels)

    private fun NumberNDArrayCore.toFloatTensor(): FloatNDArray {
        return when (this.type) {
            DataType.FLOAT -> this as FloatNDArray
            DataType.DOUBLE -> {
                this as DoubleNDArray
                val tiledArray = FloatTiledArray(this.linearSize, this.array.blockSize)
                for (blockIdx in this.array.indices) {
                    val inputBlock = this.array.getBlock(blockIdx)
                    val outputBlock = tiledArray.getBlock(blockIdx)

                    for (j in outputBlock.indices) {
                        outputBlock[j] = inputBlock[j].toFloat()
                    }
                }
                FloatNDArray(tiledArray, strides)
            }

            DataType.INT -> {
                this as IntNDArray
                val tiledArray = FloatTiledArray(this.linearSize, this.array.blockSize)
                for (blockIdx in this.array.indices) {
                    val inputBlock = this.array.getBlock(blockIdx)
                    val outputBlock = tiledArray.getBlock(blockIdx)

                    for (j in outputBlock.indices) {
                        outputBlock[j] = inputBlock[j].toFloat()
                    }
                }
                FloatNDArray(tiledArray, strides)
            }
            DataType.LONG -> {
                this as LongNDArray
                val tiledArray = FloatTiledArray(this.linearSize, this.array.blockSize)
                for (blockIdx in this.array.indices) {
                    val inputBlock = this.array.getBlock(blockIdx)
                    val outputBlock = tiledArray.getBlock(blockIdx)

                    for (j in outputBlock.indices) {
                        outputBlock[j] = inputBlock[j].toFloat()
                    }
                }
                FloatNDArray(tiledArray, strides)
            }

            else -> error("Unsupported data type, SVMClassifier supports only FLOAT, DOUBLE, INT and LONG inputs")
        }
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        // [batch, features_count] or [features_count], reshape to 2D
        val input = (inputs[0]!!.data as NumberNDArrayCore).let {
            if (it.rank == 1) it.reshape(intArrayOf(1, it.shape[0])) else it
        }.toFloatTensor()

        val (labels, scores) = svm.run(input)


        return listOf(labels.asTensor("Y"), scores.asTensor("Z"))
    }
}
