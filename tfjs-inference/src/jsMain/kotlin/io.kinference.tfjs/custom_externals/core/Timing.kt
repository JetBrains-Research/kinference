@file:JsModule("@tensorflow/tfjs-core")

package io.kinference.tfjs.custom_externals.core

import kotlin.js.Promise

external fun time(f: () -> Unit): dynamic

external fun profile(f: () -> Unit): Promise<ProfileInfo>


