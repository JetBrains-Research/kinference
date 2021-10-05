package io.kinference.custom_externals.extensions

import io.kinference.custom_externals.core.TensorTFJS
import io.kinference.custom_externals.core.tidy

fun tidy(fn: () -> Array<TensorTFJS>): Array<TensorTFJS> = tidy(fn, null)
