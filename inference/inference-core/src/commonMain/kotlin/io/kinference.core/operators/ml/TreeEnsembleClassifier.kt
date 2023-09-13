package io.kinference.core.operators.ml

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.operators.ml.trees.*
import io.kinference.core.operators.ml.utils.LabelsInfo
import io.kinference.core.operators.ml.utils.LabelsInfo.Companion.getLabelsInfo
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto.AttributeType
import io.kinference.protobuf.message.TensorProto
import io.kinference.trees.TreeEnsembleInfo

sealed class TreeEnsembleClassifier(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): TreeEnsembleClassifier {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in TreeEnsembleClassifierVer1.VERSION.asRange() -> TreeEnsembleClassifierVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of TreeEnsembleClassifier operator: $version")
            }
        }
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

        val labelsIntAttributeInfo = AttributeInfo("classlabels_int64s", setOf(AttributeType.INTS), required = false)
        val labelsStringAttributeInfo = AttributeInfo("classlabels_strings", setOf(AttributeType.STRINGS), required = false)


        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("base_values", setOf(AttributeType.FLOATS), required = false),
            AttributeInfo("class_ids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("class_nodeids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("class_treeids", setOf(AttributeType.INTS), required = true),
            AttributeInfo("class_weights", setOf(AttributeType.FLOATS), required = true),
            labelsIntAttributeInfo,
            labelsStringAttributeInfo,
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

        private fun writeLabels(dataType: TensorProto.DataType, shape: IntArray, write: (Int) -> Any): NDArray {
            return when (dataType) {
                TensorProto.DataType.INT64 -> LongNDArray(shape) { write(it) as Long }
                TensorProto.DataType.STRING -> StringNDArray(shape) { write(it) as String }
                else -> error("Unsupported data type: $dataType")
            }
        }
    }

    private val labels = getLabelsInfo(
        intLabelsName = labelsIntAttributeInfo.name,
        stringLabelsName = labelsStringAttributeInfo.name
    )


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

    private val treeEnsemble = KICoreTreeEnsemble.fromInfo(ensembleInfo)

    private suspend fun labeledTopClasses(array: FloatNDArray): NDArray {
        val shape = intArrayOf(array.shape[0])
        val labelsIndices = array.argmax(axis = -1).array.pointer()
        return writeLabels(labels.labelsDataType, shape) {
            labels.labels[labelsIndices.getAndIncrement()]!!
        }
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val inputData = inputs[0]!!.data as NumberNDArrayCore
        val classScores = treeEnsemble.execute(inputData)
        val classLabels = labeledTopClasses(classScores)
        return listOf(classLabels.asTensor("Y"), classScores.asTensor("Z"))
    }
}
