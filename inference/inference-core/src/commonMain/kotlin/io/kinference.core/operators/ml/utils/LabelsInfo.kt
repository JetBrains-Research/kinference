package io.kinference.core.operators.ml.utils

import io.kinference.operator.Operator
import io.kinference.protobuf.message.TensorProto

internal sealed class LabelsInfo<T>(val labels: List<T>, val labelsDataType: TensorProto.DataType) {
    val size: Int = labels.size

    class LongLabelsInfo(labels: List<Long>) : LabelsInfo<Long>(labels, TensorProto.DataType.INT64)
    class StringLabelsInfo(labels: List<String>) : LabelsInfo<String>(labels, TensorProto.DataType.STRING)

    companion object {
        internal fun Operator<*, *>.getLabelsInfo(
            intLabelsName: String = "classlabels_int64s",
            stringLabelsName: String = "classlabels_strings"
        ): LabelsInfo<*> {
            return if (hasAttributeSet(intLabelsName)) {
                val attr = getAttribute<LongArray>(intLabelsName)
                LongLabelsInfo(attr.toList())
            } else {
                require(hasAttributeSet(stringLabelsName)) { "Either $intLabelsName or $stringLabelsName attribute should be specified" }
                val attr = getAttribute<List<String>>(stringLabelsName)
                StringLabelsInfo(attr)
            }
        }
    }
}
