package io.kinference.core.data

import io.kinference.core.types.ValueInfo
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType

abstract class KIONNXData<T>(override val type: ONNXDataType, override val data: T, val info: ValueInfo): ONNXData<T> {
    override val name: String?
        get() = info.name
}
