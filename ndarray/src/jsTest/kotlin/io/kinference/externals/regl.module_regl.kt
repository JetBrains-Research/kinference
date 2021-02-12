@file:Suppress("INTERFACE_WITH_SUPERCLASS", "OVERRIDING_FINAL_MEMBER", "RETURN_TYPE_MISMATCH_ON_OVERRIDE", "CONFLICTING_OVERLOADS")

package io.kinference.externals

import kotlin.js.*
import org.khronos.webgl.*
import org.w3c.dom.*

@JsModule("regl")
external fun REGL(): Regl

@JsModule("regl")
external fun REGL(selector: String): Regl

@JsModule("regl")
external fun REGL(container: HTMLElement): Regl

@JsModule("regl")
external fun REGL(canvas: HTMLCanvasElement): Regl

@JsModule("regl")
external fun REGL(gl: WebGLRenderingContext): Regl

@JsModule("regl")
external fun REGL(options: InitializationOptions): Regl
