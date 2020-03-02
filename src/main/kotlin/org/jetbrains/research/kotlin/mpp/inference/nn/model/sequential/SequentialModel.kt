package org.jetbrains.research.kotlin.mpp.inference.nn.model.sequential

import org.jetbrains.research.kotlin.mpp.inference.nn.layer.Layer
import org.jetbrains.research.kotlin.mpp.inference.nn.model.Model

abstract class SequentialModel<in T, out V>(
    name: String,
    protected open val layers: List<Layer<*>>,
    val batchInputShape: List<Int?>? = null
) : Model<T, V>(name)
