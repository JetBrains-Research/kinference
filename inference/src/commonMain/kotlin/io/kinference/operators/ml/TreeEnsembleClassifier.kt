package io.kinference.operators.ml

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ProfilingContext
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.onnx.AttributeProto.AttributeType
import io.kinference.onnx.TensorProto
import io.kinference.operators.*
import io.kinference.operators.ml.TreeEnsembleOperator.Companion.toFloatNDArray
import io.kinference.operators.ml.trees.TreeEnsembleBuilder
import kotlin.time.ExperimentalTime

@ExperimentalTime
class TreeEnsembleClassifier(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {
    companion object {
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

        private val INFO = OperatorInfo("TreeEnsembleClassifier", ATTRIBUTES_INFO, TreeEnsembleOperator.INPUTS_INFO, OUTPUTS_INFO)

        fun FloatNDArray.maxIdx(): Int {
            var max = Float.MIN_VALUE
            var maxIndex = 0
            var offset = 0
            for (block in array.blocks) {
                for (idx in block.indices) {
                    val tmp = block[idx]
                    if (tmp > max) {
                        max = tmp
                        maxIndex = idx + offset
                    }
                }
                offset += block.size
            }

            return maxIndex
        }

        private fun writeLabels(dataType: TensorProto.DataType, shape: IntArray, write: (Int) -> Any): NDArray {
            return when (dataType) {
                TensorProto.DataType.INT64 -> LongNDArray(shape) { write(it) as Long }
                TensorProto.DataType.STRING -> StringNDArray(shape) { write(it) as String }
                else -> error("Unsupported data type: $dataType")
            }
        }
    }

    @Suppress("PropertyName")
    class ClassifierInfo(map: Map<String, Any?>) : TreeEnsembleOperator.BaseEnsembleInfo(map) {
        val class_ids: List<Number> by map
        val class_nodeids: List<Number> by map
        val class_treeids: List<Number> by map
        val class_weights: List<Number> by map
        val classlabels_int64s: List<Number>? by map
        val classlabels_strings: List<String>? by map
    }

    private val ensembleInfo: ClassifierInfo
        get() = ClassifierInfo(this.attributes.mapValues { it.value.value }.withDefault { null })

    private val ensemble = TreeEnsembleBuilder.fromInfo(ensembleInfo)

    private fun labeledTopClasses(array: FloatNDArray): NDArray {
        val rows = array.rows
        val shape = intArrayOf(array.shape[0])
        return writeLabels(ensemble.labelsInfo!!.labelsDataType, shape) {
            ensemble.labelsInfo.labels[(rows[it] as FloatNDArray).maxIdx()]
        }
    }

    override fun apply(context: Context, inputs: List<Tensor?>, profilingContext: ProfilingContext?): List<Tensor?> {
        val inputData = inputs[0]!!.data.toFloatNDArray()
        val classScores = ensemble.execute(inputData) as FloatNDArray
        val classLabels = labeledTopClasses(classScores)
        return listOf(classLabels.asTensor("Y"), classScores.asTensor("Z"))
    }
}
