package io.kinference.protobuf.message

import io.kinference.protobuf.ProtobufReader

class TypeProto(
    val denotation: String? = null,
    val tensorType: Tensor? = null,
    val sequenceType: Sequence? = null,
    val mapType: Map? = null
) {
    companion object {
        fun decode(reader: ProtobufReader): TypeProto {
            var denotation: String? = null
            var tensorType: Tensor? = null
            var sequenceType: Sequence? = null
            var mapType: Map? = null
            reader.forEachTag { tag ->
                when (ReaderTag.fromInt(tag)) {
                    ReaderTag.TENSOR_TYPE -> tensorType = Tensor.decode(reader)
                    ReaderTag.SEQ_TYPE -> sequenceType = Sequence.decode(reader)
                    ReaderTag.MAP_TYPE -> mapType = Map.decode(reader)
                    ReaderTag.DENOTATION -> denotation = reader.readString()
                }
            }
            return TypeProto(denotation, tensorType, sequenceType, mapType)
        }
    }

    private enum class ReaderTag(val tag: Int) {
        TENSOR_TYPE(1),
        SEQ_TYPE(4),
        MAP_TYPE(5),
        DENOTATION(6);

        companion object {
            fun fromInt(tag: Int) = values().first { it.tag == tag }
        }
    }

    class Tensor(val elem_type: Int? = null, val shape: TensorShapeProto? = null) {
        companion object {
            fun decode(reader: ProtobufReader): Tensor {
                var elemType: Int? = null
                var shape: TensorShapeProto? = null
                reader.forEachTag { tag ->
                    when (ReaderTag.fromInt(tag)) {
                        ReaderTag.ELEMENT_TYPE -> elemType = reader.readInt()
                        ReaderTag.SHAPE -> shape = TensorShapeProto.decode(reader)
                        else -> reader.readUnknownField(tag)
                    }
                }
                return Tensor(elemType, shape)
            }
        }

        private enum class ReaderTag(val tag: Int) {
            ELEMENT_TYPE(1),
            SHAPE(2);

            companion object {
                fun fromInt(tag: Int) = values().first { it.tag == tag }
            }
        }
    }

    class Sequence(val elem_type: TypeProto? = null) {
        companion object {
            fun decode(reader: ProtobufReader): Sequence {
                var elemType: TypeProto? = null
                reader.forEachTag { tag ->
                    if (tag != 1) error("Unexpected tag $tag")
                    elemType = TypeProto.decode(reader)
                }
                return Sequence(elemType)
            }
        }
    }

    class Map(val key_type: Int? = null, val value_type: TypeProto? = null) {
        companion object {
            fun decode(reader: ProtobufReader): Map {
                var keyType: Int? = null
                var valueType: TypeProto? = null
                reader.forEachTag { tag ->
                    when (ReaderTag.fromInt(tag)) {
                        ReaderTag.KEY_TYPE -> keyType = reader.readInt()
                        ReaderTag.VALUE_TYPE -> valueType = TypeProto.decode(reader)
                    }
                }
                return Map(keyType, valueType)
            }
        }

        private enum class ReaderTag(val tag: Int) {
            KEY_TYPE(1),
            VALUE_TYPE(2);

            companion object {
                fun fromInt(tag: Int) = values().first { it.tag == tag }
            }
        }
    }
}
