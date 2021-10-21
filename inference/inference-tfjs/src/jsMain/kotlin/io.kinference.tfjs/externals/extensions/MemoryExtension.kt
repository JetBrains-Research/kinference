package io.kinference.tfjs.externals.extensions

import io.kinference.tfjs.externals.core.NDArrayTFJS
import io.kinference.tfjs.externals.core.tidy

fun tidy(fn: () -> Array<NDArrayTFJS>): Array<NDArrayTFJS> = tidy(fn, null)
