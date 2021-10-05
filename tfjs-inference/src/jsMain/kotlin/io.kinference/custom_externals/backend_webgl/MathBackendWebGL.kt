@file:JsModule("@tensorflow/tfjs-backend-webgl")
package io.kinference.custom_externals.backend_webgl

import io.kinference.custom_externals.core.KernelBackend

open external class MathBackendWebGL(gpgpu: dynamic) : KernelBackend
