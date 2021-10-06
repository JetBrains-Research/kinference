package io.kinference.tfjs.custom_externals.extensions

import io.kinference.tfjs.custom_externals.core.TensorTFJS
import io.kinference.tfjs.custom_externals.core.tidy

fun tidy(fn: () -> Array<TensorTFJS>): Array<TensorTFJS> = tidy(fn, null)
