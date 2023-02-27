package io.kinference.core.operators.ml

import io.kinference.core.KIONNXData
import io.kinference.attribute.Attribute
import io.kinference.core.data.map.KIONNXMap
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.pointers.FloatPointer
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.types.*
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import kotlin.collections.HashMap
import kotlin.time.ExperimentalTime

sealed class ZipMap(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KITensor, KIONNXSequence>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in ZipMapVer1.VERSION.asRange() -> ZipMapVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of ZipMap operator: $version")
        }
    }
}

@ExperimentalTime
class ZipMapVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : ZipMap(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val OUT_TYPE_CONSTRAINTS = setOf(TensorProto.DataType.INT64, TensorProto.DataType.STRING, TensorProto.DataType.FLOAT)

        private val INPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.FLOAT), "X", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, OUT_TYPE_CONSTRAINTS, "Z", optional = false)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("classlabels_int64s", setOf(AttributeProto.AttributeType.INTS), required = false),
            AttributeInfo("classlabels_strings", setOf(AttributeProto.AttributeType.STRINGS), required = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("ZipMap", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "ai.onnx.ml")

        private fun <T : Any> FloatNDArray.asSeqWithLabels(labels: Labels<T>, mapInfo: ValueTypeInfo.MapTypeInfo): KIONNXSequence {
            val seqInfo = ValueTypeInfo.SequenceTypeInfo(mapInfo)
            val rows = if (rank == 1) 1 else shape[0]
            val columns = shape.last()

            val inputPointer = FloatPointer(array)
            return KIONNXSequence("Z", seqInfo, rows) {
                val map = HashMap<T, KIONNXData<*>>(columns)
                repeat(columns) {
                    val value = inputPointer.getAndIncrement()
                    val tensor = FloatNDArray.scalar(value).asTensor()
                    map[labels[it]] = tensor
                }
                KIONNXMap(null, map as Map<Any, KIONNXData<*>>, mapInfo)
            }
        }
    }

    sealed class Labels<T> {
        abstract operator fun get(i: Int): T

        class StringLabels(private val labels: List<String>): Labels<String>() {
            override fun get(i: Int): String = labels[i]
        }

        class LongLabels(private val labels: LongArray): Labels<Long>() {
            override fun get(i: Int): Long = labels[i]
        }
    }

    private val classLabelsLong: Labels.LongLabels? by attributeOrNull("classlabels_int64s") { labels: LongArray? -> Labels.LongLabels(labels!!) }
    private val classLabelsString: Labels.StringLabels? by attributeOrNull("classlabels_strings") { labels: List<String>? -> Labels.StringLabels(labels!!) }

    private val outputMapInfo: ValueTypeInfo.MapTypeInfo
        get() {
            val mapKeyType = if (classLabelsLong != null) TensorProto.DataType.INT64 else TensorProto.DataType.STRING
            val mapValueInfo = ValueTypeInfo.TensorTypeInfo(TensorShape.empty(), TensorProto.DataType.FLOAT)
            return ValueTypeInfo.MapTypeInfo(mapKeyType, mapValueInfo)
        }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KIONNXSequence?> {
        val labels = classLabelsLong ?: classLabelsString
        requireNotNull(labels) { "Class labels should be specified" }

        val input = inputs[0]!!.data as FloatNDArray
        require(input.rank == 2)

        return listOf(input.asSeqWithLabels(labels, outputMapInfo))
    }
}
