package io.kinference.protobuf.message

import com.squareup.wire.*

enum class OperatorStatus(override val value: Int) : WireEnum {
    EXPERIMENTAL(0),
    STABLE(1);

    companion object {
        val ADAPTER: ProtoAdapter<OperatorStatus> = object : EnumAdapter<OperatorStatus>(OperatorStatus::class) {
            override fun fromValue(value: Int): OperatorStatus = OperatorStatus.fromValue(value)
        }

        fun fromValue(value: Int): OperatorStatus = when (value) {
            0 -> EXPERIMENTAL
            1 -> STABLE
            else -> error("Cannot convert from value $value")
        }
    }
}
