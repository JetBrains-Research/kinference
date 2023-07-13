package io.kinference.tfjs.operators.ml

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.dataInt
import io.kinference.ndarray.extensions.tidyNDArrays
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto.AttributeType
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.operators.ml.trees.TFJSTreeEnsemble
import io.kinference.trees.TreeEnsembleInfo

sealed class TreeEnsembleClassifier(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): TreeEnsembleClassifier {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in TreeEnsembleClassifierVer1.VERSION.asRange() -> TreeEnsembleClassifierVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of TreeEnsembleClassifier operator: $version")
            }
        }
    }

    sealed class LabelsInfo<T>(val labels: Array<T>, val labelsDataType: TensorProto.DataType) {
        val size: Int = labels.size

        class LongLabelsInfo(labels: Array<Long>) : LabelsInfo<Long>(labels, TensorProto.DataType.INT64)
        class StringLabelsInfo(labels: Array<String>) : LabelsInfo<String>(labels, TensorProto.DataType.STRING)
    }
}


class TreeEnsembleClassifierVer1(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : TreeEnsembleClassifier(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            //TensorProto.DataType.INT32,
            //TensorProto.DataType.INT64,
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

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("base_values", setOf(AttributeType.FLOATS), required = false),
            AttributeInfo("class_ids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("class_nodeids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("class_treeids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("class_weights", setOf(AttributeType.FLOATS), required = true),
            AttributeInfo("classlabels_int64s", setOf(AttributeType.INTS), required = false),
            AttributeInfo("classlabels_strings", setOf(AttributeType.STRINGS), required = false),
            AttributeInfo("nodes_falsenodeids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("nodes_featureids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("nodes_hitrates", setOf(AttributeType.FLOATS), required = false),
            AttributeInfo("nodes_missing_value_tracks_true", setOf(AttributeType.INTS), required = false),
            AttributeInfo("nodes_modes", setOf(AttributeType.STRINGS), required = true),
            AttributeInfo("nodes_nodeids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("nodes_treeids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("nodes_truenodeids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("nodes_values", setOf(AttributeType.FLOATS), required = true),
            AttributeInfo("post_transform", setOf(AttributeType.STRING), required = false, default = "NONE"),
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("TreeEnsembleClassifier", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.ML_DOMAIN)

        private fun writeLabels(dataType: TensorProto.DataType, shape: Array<Int>, write: (Int) -> Any): NDArrayTFJS {
            return when (dataType) {
                TensorProto.DataType.INT64 -> NDArrayTFJS.int(shape) { it: Int -> (write(it) as Long).toInt() }
                TensorProto.DataType.STRING -> NDArrayTFJS.string(shape) { it: Int -> (write(it) as String) }
                else -> error("Unsupported data type: $dataType")
            }
        }
    }

    private val labels: LabelsInfo<*> = if (hasAttributeSet("classlabels_int64s")) {
        val attr = getAttribute<LongArray>("classlabels_int64s")
        LabelsInfo.LongLabelsInfo(attr.toTypedArray())
    } else {
        require(hasAttributeSet("classlabels_strings")) { "Either classlabels_int64s or classlabels_strings attribute should be specified" }
        val attr = getAttribute<List<String>>("classlabels_strings")
        LabelsInfo.StringLabelsInfo(attr.toTypedArray())
    }

    private val ensembleInfo = TreeEnsembleInfo(
        baseValues = getAttributeOrNull("base_values"),
        featureIds = getAttribute("nodes_featureids"),
        nodeModes = getAttribute("nodes_modes"),
        nodeIds = getAttribute("nodes_nodeids"),
        treeIds = getAttribute("nodes_treeids"),
        falseNodeIds = getAttribute("nodes_falsenodeids"),
        trueNodeIds = getAttribute("nodes_truenodeids"),
        nodeValues = getAttribute("nodes_values"),
        postTransform = getAttributeOrNull("post_transform"),
        aggregator = null,
        targetIds = getAttribute("class_ids"),
        targetNodeIds = getAttribute("class_nodeids"),
        targetNodeTreeIds = getAttribute("class_treeids"),
        targetWeights = getAttribute("class_weights"),
        numTargets = labels.size
    )

    private val ensemble = TFJSTreeEnsemble.fromInfo(ensembleInfo)

    private suspend fun labeledTopClasses(array: NumberNDArrayTFJS): NDArrayTFJS {
        val shape = arrayOf(array.shape[0])
        val labelsIndices = array.argmax(axis = -1).dataInt()
        return writeLabels(labels.labelsDataType, shape) {
            labels.labels[labelsIndices[it]]!!
        }
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val (labels, scores) = tidyNDArrays {
            val inputData = inputs[0]!!.data as NumberNDArrayTFJS
            val classScores = ensemble.execute(inputData)
            val classLabels = labeledTopClasses(classScores)
            arrayOf(classLabels, classScores)
        }
        return listOf(labels.asTensor("Y"), scores.asTensor("Z"))
    }
}
