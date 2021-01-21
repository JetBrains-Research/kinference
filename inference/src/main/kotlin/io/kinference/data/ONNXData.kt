package io.kinference.data

import io.kinference.types.ValueInfo

abstract class ONNXData(val type: ONNXDataType, open val info: ValueInfo) {
    abstract fun rename(name: String): ONNXData
}
