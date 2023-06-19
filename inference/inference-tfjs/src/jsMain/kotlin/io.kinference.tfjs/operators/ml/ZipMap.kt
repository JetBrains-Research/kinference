package io.kinference.tfjs.operators.ml

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.dataFloat
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.map.TFJSMap
import io.kinference.tfjs.data.seq.TFJSSequence
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.types.TensorShape
import io.kinference.types.ValueTypeInfo

sealed class ZipMap(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSSequence>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): ZipMap {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in ZipMapVer1.VERSION.asRange() -> ZipMapVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of ZipMap operator: $version")
            }
        }
    }
}


class ZipMapVer1(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : ZipMap(name, INFO, attributes, inputs, outputs) {
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
        private val INFO = OperatorInfo("ZipMap", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.ML_DOMAIN)

        private fun <T : Any> NumberNDArrayTFJS.asSeqWithLabels(labels: Labels<T>, mapInfo: ValueTypeInfo.MapTypeInfo): TFJSSequence {
            val seqInfo = ValueTypeInfo.SequenceTypeInfo(mapInfo)
            val rows = if (rank == 1) 1 else shape[0]
            val columns = shape.last()

            val inputArray = this.dataFloat()
            return TFJSSequence("Z", seqInfo, rows) {
                val map = HashMap<T, TFJSData<*>>(columns)
                val offset = it * columns
                repeat(columns) { col ->
                    val value = inputArray[offset + col]
                    val tensor = NDArrayTFJS.floatScalar(value).asTensor()
                    map[labels[it]] = tensor
                }
                TFJSMap(null, map as Map<Any, TFJSData<*>>, mapInfo)
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

    private val classLabelsLong: Labels.LongLabels? by attributeOrNull("classlabels_int64s") { labels: LongArray? -> labels?.let { Labels.LongLabels(labels) } }
    private val classLabelsString: Labels.StringLabels? by attributeOrNull("classlabels_strings") { labels: List<String>? -> labels?.let { Labels.StringLabels(labels) } }

    private val outputMapInfo: ValueTypeInfo.MapTypeInfo
        get() {
            val mapKeyType = if (classLabelsLong != null) TensorProto.DataType.INT64 else TensorProto.DataType.STRING
            val mapValueInfo = ValueTypeInfo.TensorTypeInfo(TensorShape.empty(), TensorProto.DataType.FLOAT)
            return ValueTypeInfo.MapTypeInfo(mapKeyType, mapValueInfo)
        }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSSequence?> {
        val labels = classLabelsLong ?: classLabelsString
        requireNotNull(labels) { "Class labels should be specified" }

        val input = inputs[0]!!.data as NumberNDArrayTFJS
        require(input.rank == 2) { "Expected input rank=2. Actual input rank=${input.rank}" }

        return listOf(input.asSeqWithLabels(labels, outputMapInfo))
    }
}
