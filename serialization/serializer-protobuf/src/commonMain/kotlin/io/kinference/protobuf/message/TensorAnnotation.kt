package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class TensorAnnotation(
    var tensorName: String? = null,
    val quantParameterTensorNames: MutableList<StringStringEntryProto> = ArrayList()
) {
    companion object {
        suspend fun decode(reader: ProtobufReader): TensorAnnotation {
            val proto = TensorAnnotation()
            reader.forEachTag { tag ->
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.TENSOR_NAME -> proto.tensorName = reader.readString()
                    ReaderTag.QUANT_PARAMS_TENSOR_NAMES -> proto.quantParameterTensorNames.add(StringStringEntryProto.decode(reader))
                    null -> reader.readUnknownField(tag)
                }
            }
            return proto
        }
    }

    private enum class ReaderTag(val tag: Int) {
        TENSOR_NAME(1),
        QUANT_PARAMS_TENSOR_NAMES(2);

        companion object {
            fun fromInt(tag: Int) = values().firstOrNull { it.tag == tag }
        }
    }
}
