@file:Suppress("INTERFACE_WITH_SUPERCLASS", "OVERRIDING_FINAL_MEMBER", "RETURN_TYPE_MISMATCH_ON_OVERRIDE", "CONFLICTING_OVERLOADS")
package io.kinference.externals

import kotlin.js.*
import org.khronos.webgl.*
import org.w3c.dom.*
import tsstdlib.Partial

external interface Regl {
    var attributes: WebGLContextAttributes
    var _gl: WebGLRenderingContext
    var limits: Limits
    var stats: Stats
    @nativeInvoke
    operator fun <Uniforms : Any, Attributes : Any, Props : Any, OwnContext : Any, ParentContext : DefaultContext> invoke(drawConfig: DrawConfig<Uniforms, Attributes, Props, OwnContext, ParentContext>): DrawCommand<ParentContext /* ParentContext & OwnContext */, Props>
    fun clear(options: ClearOptions)
    fun <T> read(): T
    fun <T> read(data: T): T
    fun <T> read(options: ReadOptions<T>): T
    fun <Key : Any> prop(name: Key): DynamicVariable<Any>
    fun <K : Any> context(name: K): DynamicVariable<Any>
    fun <Key : Any> `this`(name: Key): DynamicVariable<Any>
    fun draw()
    fun buffer(length: Number): Buffer
    fun buffer(data: Array<Number>): Buffer
    fun buffer(data: Array<Array<Number>>): Buffer
    fun buffer(data: Uint8Array): Buffer
    fun buffer(data: Int8Array): Buffer
    fun buffer(data: Uint16Array): Buffer
    fun buffer(data: Int16Array): Buffer
    fun buffer(data: Uint32Array): Buffer
    fun buffer(data: Int32Array): Buffer
    fun buffer(data: Float32Array): Buffer
    fun buffer(options: BufferOptions): Buffer
    fun elements(data: Array<Number>): Elements
    fun elements(data: Array<Array<Number>>): Elements
    fun elements(data: Uint8Array): Elements
    fun elements(data: Uint16Array): Elements
    fun elements(data: Uint32Array): Elements
    fun elements(options: ElementsOptions): Elements
    fun texture(): Texture2D
    fun texture(radius: Number): Texture2D
    fun texture(width: Number, height: Number): Texture2D
    fun texture(data: Array<Number>): Texture2D
    fun texture(data: Array<Array<Number>>): Texture2D
    fun texture(data: Array<Array<Array<Number>>>): Texture2D
    fun texture(data: ArrayBufferView): Texture2D
    fun texture(data: NDArrayLike): Texture2D
    fun texture(data: HTMLImageElement): Texture2D
    fun texture(data: HTMLVideoElement): Texture2D
    fun texture(data: HTMLCanvasElement): Texture2D
    fun texture(data: CanvasRenderingContext2D): Texture2D
    fun texture(options: Texture2DOptions): Texture2D
    fun cube(): TextureCube
    fun cube(radius: Number): TextureCube
    fun cube(posXData: Array<Number>, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    fun cube(posXData: Array<Array<Number>>, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    fun cube(posXData: Array<Array<Array<Number>>>, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    fun cube(posXData: ArrayBufferView, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    fun cube(posXData: NDArrayLike, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    fun cube(posXData: HTMLImageElement, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    fun cube(posXData: HTMLVideoElement, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    fun cube(posXData: HTMLCanvasElement, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    fun cube(posXData: CanvasRenderingContext2D, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    fun cube(posXOptions: Texture2DOptions, negXOptions: Texture2DOptions, posYOptions: Texture2DOptions, negYOptions: Texture2DOptions, posZOptions: Texture2DOptions, negZOptions: Texture2DOptions): TextureCube
    fun cube(options: TextureCubeOptions): TextureCube
    fun renderbuffer(): Renderbuffer
    fun renderbuffer(radius: Number): Renderbuffer
    fun renderbuffer(width: Number, height: Number): Renderbuffer
    fun renderbuffer(options: RenderbufferOptions): Renderbuffer
    fun framebuffer(): Framebuffer2D
    fun framebuffer(radius: Number): Framebuffer2D
    fun framebuffer(width: Number, height: Number): Framebuffer2D
    fun framebuffer(options: FramebufferOptions): Framebuffer2D
    fun framebufferCube(): FramebufferCube
    fun framebufferCube(radius: Number): FramebufferCube
    fun framebufferCube(options: FramebufferCubeOptions): FramebufferCube
    fun vao(attributes: Array<Any /* ConstantAttribute | AttributeConfig | io.kinference.externals.Buffer | Array<Number> | Array<Array<Number>> | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array | Float32Array */>): VertexArrayObject
    fun vao(attributes: VertexArrayOptions): VertexArrayObject
    fun frame(callback: FrameCallback): Cancellable
    fun on(type: String /* "frame" */, callback: FrameCallback): Cancellable
    fun on(type: String /* "lost" | "restore" | "destroy" */, callback: () -> Unit): Cancellable
    fun hasExtension(name: String): Boolean
    fun poll()
    fun now(): Number
    fun destroy()
    fun _refresh()
}

external interface InitializationOptions {
    var gl: WebGLRenderingContext?
        get() = definedExternally
        set(value) = definedExternally
    var canvas: dynamic /* String? | HTMLCanvasElement? */
        get() = definedExternally
        set(value) = definedExternally
    var container: dynamic /* String? | HTMLElement? */
        get() = definedExternally
        set(value) = definedExternally
    var attributes: WebGLContextAttributes?
        get() = definedExternally
        set(value) = definedExternally
    var pixelRatio: Number?
        get() = definedExternally
        set(value) = definedExternally
    var extensions: dynamic /* String? | Array<String>? */
        get() = definedExternally
        set(value) = definedExternally
    var optionalExtensions: dynamic /* String? | Array<String>? */
        get() = definedExternally
        set(value) = definedExternally
    var profile: Boolean?
        get() = definedExternally
        set(value) = definedExternally
    var onDone: ((err: Error?, regl: Regl) -> Unit)?
        get() = definedExternally
        set(value) = definedExternally
}

external interface DefaultContext {
    var tick: Number
    var time: Number
    var viewportWidth: Number
    var viewportHeight: Number
    var drawingBufferWidth: Number
    var drawingBufferHeight: Number
    var pixelRatio: Number
}

typealias UserContext<ParentContext, OwnContext, Props> = Any

external interface Cancellable {
    fun cancel()
}

typealias FrameCallback = (context: DefaultContext) -> Unit

external interface DynamicVariable<Return> {
    var id: Number
    var type: Number
    var data: String
}

typealias DynamicVariableFn<Return, Context, Props> = (context: Context, props: Props, batchId: Number) -> Return

typealias MaybeNestedDynamic<Type, Context, Props> = Any

external interface ClearOptions {
    var color: dynamic /* JsTuple<Number, Number, Number, Number> */
        get() = definedExternally
        set(value) = definedExternally
    var depth: Number?
        get() = definedExternally
        set(value) = definedExternally
    var stencil: Number?
        get() = definedExternally
        set(value) = definedExternally
    var framebuffer: dynamic /* io.kinference.externals.Framebuffer2D? | io.kinference.externals.FramebufferCube? */
        get() = definedExternally
        set(value) = definedExternally
}

external interface ReadOptions<T> {
    var data: T?
        get() = definedExternally
        set(value) = definedExternally
    var x: Number?
        get() = definedExternally
        set(value) = definedExternally
    var y: Number?
        get() = definedExternally
        set(value) = definedExternally
    var width: Number?
        get() = definedExternally
        set(value) = definedExternally
    var height: Number?
        get() = definedExternally
        set(value) = definedExternally
    var framebuffer: dynamic /* io.kinference.externals.Framebuffer2D? | io.kinference.externals.FramebufferCube? */
        get() = definedExternally
        set(value) = definedExternally
}

typealias CommandBodyFn<Context, Props> = (context: Context, props: Props, batchId: Number) -> Unit

external interface DrawCommand<Context : DefaultContext, Props : Any> {
    var stats: CommandStats
    @nativeInvoke
    operator fun invoke(body: CommandBodyFn<Context, Props> = definedExternally)
    @nativeInvoke
    operator fun invoke()
    @nativeInvoke
    operator fun invoke(count: Number, body: CommandBodyFn<Context, Props> = definedExternally)
    @nativeInvoke
    operator fun invoke(count: Number)
    @nativeInvoke
    operator fun invoke(props: Partial<Props>, body: CommandBodyFn<Context, Props> = definedExternally)
    @nativeInvoke
    operator fun invoke(props: Partial<Props>)
    @nativeInvoke
    operator fun invoke(props: Array<Partial<Props>>, body: CommandBodyFn<Context, Props> = definedExternally)
    @nativeInvoke
    operator fun invoke(props: Array<Partial<Props>>)
}

external interface DrawConfig<Uniforms : Any, Attributes : Any, Props : Any, OwnContext : Any, ParentContext : DefaultContext> {
    var vert: dynamic /* String? | DynamicVariable<String>? | DynamicVariableFn<String, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var frag: dynamic /* String? | DynamicVariable<String>? | DynamicVariableFn<String, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var context: UserContext<ParentContext, OwnContext, Props>?
        get() = definedExternally
        set(value) = definedExternally
    var uniforms: MaybeDynamicUniforms<Uniforms, ParentContext /* ParentContext & OwnContext */, Props>?
        get() = definedExternally
        set(value) = definedExternally
    var attributes: MaybeDynamicAttributes<Attributes, ParentContext /* ParentContext & OwnContext */, Props>?
        get() = definedExternally
        set(value) = definedExternally
    var vao: dynamic /* io.kinference.externals.VertexArrayObject? | Array<dynamic /* ConstantAttribute | AttributeConfig | io.kinference.externals.Buffer | Array<Number> | Array<Array<Number>> | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array | Float32Array */>? | DynamicVariable<dynamic /* io.kinference.externals.VertexArrayObject | Array<dynamic /* ConstantAttribute | AttributeConfig | io.kinference.externals.Buffer | Array<Number> | Array<Array<Number>> | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array | Float32Array */> */>? | DynamicVariableFn<dynamic /* io.kinference.externals.VertexArrayObject | Array<dynamic /* ConstantAttribute | AttributeConfig | io.kinference.externals.Buffer | Array<Number> | Array<Array<Number>> | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array | Float32Array */> */, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var primitive: dynamic /* "points" | "lines" | "line strip" | "line loop" | "triangles" | "triangle strip" | "triangle fan" | DynamicVariable<String /* "points" | "lines" | "line strip" | "line loop" | "triangles" | "triangle strip" | "triangle fan" */>? | DynamicVariableFn<String /* "points" | "lines" | "line strip" | "line loop" | "triangles" | "triangle strip" | "triangle fan" */, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var count: dynamic /* Number? | DynamicVariable<Number>? | DynamicVariableFn<Number, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var offset: dynamic /* Number? | DynamicVariable<Number>? | DynamicVariableFn<Number, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var instances: dynamic /* Number? | DynamicVariable<Number>? | DynamicVariableFn<Number, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var elements: dynamic /* io.kinference.externals.Elements? | Array<Number>? | Array<Array<Number>>? | Uint8Array? | Uint16Array? | Uint32Array? | ElementsOptions? | DynamicVariable<dynamic /* io.kinference.externals.Elements? | Array<Number>? | Array<Array<Number>>? | Uint8Array? | Uint16Array? | Uint32Array? | ElementsOptions? */>? | DynamicVariableFn<dynamic /* io.kinference.externals.Elements? | Array<Number>? | Array<Array<Number>>? | Uint8Array? | Uint16Array? | Uint32Array? | ElementsOptions? */, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var framebuffer: dynamic /* io.kinference.externals.Framebuffer2D? | io.kinference.externals.FramebufferCube? | DynamicVariable<dynamic /* io.kinference.externals.Framebuffer2D? | io.kinference.externals.FramebufferCube? */>? | DynamicVariableFn<dynamic /* io.kinference.externals.Framebuffer2D? | io.kinference.externals.FramebufferCube? */, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var profile: dynamic /* Boolean? | DynamicVariable<Boolean>? | DynamicVariableFn<Boolean, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var depth: dynamic /* io.kinference.externals.DepthTestOptions? | DynamicVariable<io.kinference.externals.DepthTestOptions>? | DynamicVariableFn<io.kinference.externals.DepthTestOptions, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var blend: dynamic /* io.kinference.externals.BlendingOptions? | DynamicVariable<io.kinference.externals.BlendingOptions>? | DynamicVariableFn<io.kinference.externals.BlendingOptions, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var stencil: dynamic /* io.kinference.externals.StencilOptions? | DynamicVariable<io.kinference.externals.StencilOptions>? | DynamicVariableFn<io.kinference.externals.StencilOptions, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var polygonOffset: dynamic /* io.kinference.externals.PolygonOffsetOptions? | DynamicVariable<io.kinference.externals.PolygonOffsetOptions>? | DynamicVariableFn<io.kinference.externals.PolygonOffsetOptions, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var cull: dynamic /* io.kinference.externals.CullingOptions? | DynamicVariable<io.kinference.externals.CullingOptions>? | DynamicVariableFn<io.kinference.externals.CullingOptions, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var frontFace: dynamic /* "cw" | "ccw" | DynamicVariable<String /* "cw" | "ccw" */>? | DynamicVariableFn<String /* "cw" | "ccw" */, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var dither: dynamic /* Boolean? | DynamicVariable<Boolean>? | DynamicVariableFn<Boolean, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var lineWidth: dynamic /* Number? | DynamicVariable<Number>? | DynamicVariableFn<Number, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var colorMask: dynamic /* JsTuple<Boolean, Boolean, Boolean, Boolean> | DynamicVariable<dynamic /* JsTuple<Boolean, Boolean, Boolean, Boolean> */>? | DynamicVariableFn<dynamic /* JsTuple<Boolean, Boolean, Boolean, Boolean> */, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var sample: dynamic /* io.kinference.externals.SamplingOptions? | DynamicVariable<io.kinference.externals.SamplingOptions>? | DynamicVariableFn<io.kinference.externals.SamplingOptions, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
    var scissor: MaybeNestedDynamic<ScissorOptions, ParentContext /* ParentContext & OwnContext */, Props>?
        get() = definedExternally
        set(value) = definedExternally
    var viewport: dynamic /* io.kinference.externals.BoundingBox? | DynamicVariable<io.kinference.externals.BoundingBox>? | DynamicVariableFn<io.kinference.externals.BoundingBox, ParentContext /* ParentContext & OwnContext */, Props>? */
        get() = definedExternally
        set(value) = definedExternally
}

external interface Uniforms {
    @nativeGetter
    operator fun get(name: String): dynamic /* Boolean? | Number? | Array<Boolean>? | Array<Number>? | Float32Array? | Int32Array? */
    @nativeSetter
    operator fun set(name: String, value: Boolean)
    @nativeSetter
    operator fun set(name: String, value: Number)
    @nativeSetter
    operator fun set(name: String, value: Array<Boolean>)
    @nativeSetter
    operator fun set(name: String, value: Array<Number>)
    @nativeSetter
    operator fun set(name: String, value: Float32Array)
    @nativeSetter
    operator fun set(name: String, value: Int32Array)
}

typealias MaybeDynamicUniforms<Uniforms, Context, Props> = Any

external interface Attributes {
    @nativeGetter
    operator fun get(name: String): dynamic /* Number? | ConstantAttribute? | AttributeConfig? | io.kinference.externals.Buffer? | Array<Number>? | Array<Array<Number>>? | Uint8Array? | Int8Array? | Uint16Array? | Int16Array? | Uint32Array? | Int32Array? | Float32Array? */
    @nativeSetter
    operator fun set(name: String, value: Number)
    @nativeSetter
    operator fun set(name: String, value: ConstantAttribute)
    @nativeSetter
    operator fun set(name: String, value: AttributeConfig)
    @nativeSetter
    operator fun set(name: String, value: Buffer)
    @nativeSetter
    operator fun set(name: String, value: Array<Number>)
    @nativeSetter
    operator fun set(name: String, value: Array<Array<Number>>)
    @nativeSetter
    operator fun set(name: String, value: Uint8Array)
    @nativeSetter
    operator fun set(name: String, value: Int8Array)
    @nativeSetter
    operator fun set(name: String, value: Uint16Array)
    @nativeSetter
    operator fun set(name: String, value: Int16Array)
    @nativeSetter
    operator fun set(name: String, value: Uint32Array)
    @nativeSetter
    operator fun set(name: String, value: Int32Array)
    @nativeSetter
    operator fun set(name: String, value: Float32Array)
}

typealias MaybeDynamicAttributes<Attributes, Context, Props> = Any

external interface ConstantAttribute {
    var constant: dynamic /* Number | Array<Number> */
        get() = definedExternally
        set(value) = definedExternally
}

external interface AttributeConfig {
    var buffer: dynamic /* io.kinference.externals.Buffer? | Boolean? */
        get() = definedExternally
        set(value) = definedExternally
    var offset: Number?
        get() = definedExternally
        set(value) = definedExternally
    var stride: Number?
        get() = definedExternally
        set(value) = definedExternally
    var normalized: Boolean?
        get() = definedExternally
        set(value) = definedExternally
    var size: Number?
        get() = definedExternally
        set(value) = definedExternally
    var divisor: Number?
        get() = definedExternally
        set(value) = definedExternally
    var type: String? /* "uint8" | "uint16" | "uint32" | "float" | "int8" | "int16" | "int32" */
        get() = definedExternally
        set(value) = definedExternally
}

external interface DepthTestOptions {
    var enable: Boolean?
        get() = definedExternally
        set(value) = definedExternally
    var mask: Boolean?
        get() = definedExternally
        set(value) = definedExternally
    var range: dynamic /* JsTuple<Number, Number> */
        get() = definedExternally
        set(value) = definedExternally
    var func: String? /* "never" | "always" | "less" | "<" | "lequal" | "<=" | "greater" | ">" | "gequal" | ">=" | "equal" | "=" | "notequal" | "!=" */
        get() = definedExternally
        set(value) = definedExternally
}

external interface BlendingOptions {
    var enable: Boolean?
        get() = definedExternally
        set(value) = definedExternally
    var equation: dynamic /* "add" | "subtract" | "reverse subtract" | "min" | "max" | io.kinference.externals.BlendingEquationSeparate? */
        get() = definedExternally
        set(value) = definedExternally
    var func: dynamic /* io.kinference.externals.BlendingFunctionCombined? | io.kinference.externals.BlendingFunctionSeparate? */
        get() = definedExternally
        set(value) = definedExternally
    var color: dynamic /* JsTuple<Number, Number, Number, Number> */
        get() = definedExternally
        set(value) = definedExternally
}

external interface BlendingEquationSeparate {
    var rgb: String /* "add" | "subtract" | "reverse subtract" | "min" | "max" */
    var alpha: String /* "add" | "subtract" | "reverse subtract" | "min" | "max" */
}

external interface BlendingFunctionCombined {
    var src: dynamic /* "zero" | 0 | "one" | 1 | "src color" | "one minus src color" | "src alpha" | "one minus src alpha" | "dst color" | "one minus dst color" | "dst alpha" | "one minus dst alpha" | "constant color" | "one minus constant color" | "constant alpha" | "one minus constant alpha" | "src alpha saturate" */
        get() = definedExternally
        set(value) = definedExternally
    var dst: dynamic /* "zero" | 0 | "one" | 1 | "src color" | "one minus src color" | "src alpha" | "one minus src alpha" | "dst color" | "one minus dst color" | "dst alpha" | "one minus dst alpha" | "constant color" | "one minus constant color" | "constant alpha" | "one minus constant alpha" | "src alpha saturate" */
        get() = definedExternally
        set(value) = definedExternally
}

external interface BlendingFunctionSeparate {
    var srcRGB: dynamic /* "zero" | 0 | "one" | 1 | "src color" | "one minus src color" | "src alpha" | "one minus src alpha" | "dst color" | "one minus dst color" | "dst alpha" | "one minus dst alpha" | "constant color" | "one minus constant color" | "constant alpha" | "one minus constant alpha" | "src alpha saturate" */
        get() = definedExternally
        set(value) = definedExternally
    var srcAlpha: dynamic /* "zero" | 0 | "one" | 1 | "src color" | "one minus src color" | "src alpha" | "one minus src alpha" | "dst color" | "one minus dst color" | "dst alpha" | "one minus dst alpha" | "constant color" | "one minus constant color" | "constant alpha" | "one minus constant alpha" | "src alpha saturate" */
        get() = definedExternally
        set(value) = definedExternally
    var dstRGB: dynamic /* "zero" | 0 | "one" | 1 | "src color" | "one minus src color" | "src alpha" | "one minus src alpha" | "dst color" | "one minus dst color" | "dst alpha" | "one minus dst alpha" | "constant color" | "one minus constant color" | "constant alpha" | "one minus constant alpha" | "src alpha saturate" */
        get() = definedExternally
        set(value) = definedExternally
    var dstAlpha: dynamic /* "zero" | 0 | "one" | 1 | "src color" | "one minus src color" | "src alpha" | "one minus src alpha" | "dst color" | "one minus dst color" | "dst alpha" | "one minus dst alpha" | "constant color" | "one minus constant color" | "constant alpha" | "one minus constant alpha" | "src alpha saturate" */
        get() = definedExternally
        set(value) = definedExternally
}

external interface StencilOptions {
    var enable: Boolean?
        get() = definedExternally
        set(value) = definedExternally
    var mask: Number?
        get() = definedExternally
        set(value) = definedExternally
    var func: StencilFunction?
        get() = definedExternally
        set(value) = definedExternally
    var opFront: StencilOperation?
        get() = definedExternally
        set(value) = definedExternally
    var opBack: StencilOperation?
        get() = definedExternally
        set(value) = definedExternally
    var op: StencilOperation?
        get() = definedExternally
        set(value) = definedExternally
}

external interface StencilFunction {
    var cmp: String /* "never" | "always" | "less" | "<" | "lequal" | "<=" | "greater" | ">" | "gequal" | ">=" | "equal" | "=" | "notequal" | "!=" */
    var ref: Number
    var mask: Number
}

external interface StencilOperation {
    var fail: String /* "zero" | "keep" | "replace" | "invert" | "increment" | "decrement" | "increment wrap" | "decrement wrap" */
    var zfail: String /* "zero" | "keep" | "replace" | "invert" | "increment" | "decrement" | "increment wrap" | "decrement wrap" */
    var zpass: String /* "zero" | "keep" | "replace" | "invert" | "increment" | "decrement" | "increment wrap" | "decrement wrap" */
}

external interface PolygonOffsetOptions {
    var enable: Boolean?
        get() = definedExternally
        set(value) = definedExternally
    var offset: PolygonOffset?
        get() = definedExternally
        set(value) = definedExternally
}

external interface PolygonOffset {
    var factor: Number
    var units: Number
}

external interface CullingOptions {
    var enable: Boolean?
        get() = definedExternally
        set(value) = definedExternally
    var face: String? /* "front" | "back" */
        get() = definedExternally
        set(value) = definedExternally
}

external interface SamplingOptions {
    var enable: Boolean?
        get() = definedExternally
        set(value) = definedExternally
    var alpha: Boolean?
        get() = definedExternally
        set(value) = definedExternally
    var coverage: SampleCoverage?
        get() = definedExternally
        set(value) = definedExternally
}

external interface SampleCoverage {
    var value: Number
    var invert: Boolean
}

external interface ScissorOptions {
    var enable: Boolean?
        get() = definedExternally
        set(value) = definedExternally
    var box: BoundingBox?
        get() = definedExternally
        set(value) = definedExternally
}

external interface BoundingBox {
    var x: Number?
        get() = definedExternally
        set(value) = definedExternally
    var y: Number?
        get() = definedExternally
        set(value) = definedExternally
    var width: Number?
        get() = definedExternally
        set(value) = definedExternally
    var height: Number?
        get() = definedExternally
        set(value) = definedExternally
}

external interface Resource {
    fun destroy()
}

external interface VertexArrayObject : Resource {
    @nativeInvoke
    operator fun invoke(attributes: Array<Any /* ConstantAttribute | AttributeConfig | io.kinference.externals.Buffer | Array<Number> | Array<Array<Number>> | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array | Float32Array */>): VertexArrayObject
    @nativeInvoke
    operator fun invoke(attributes: VertexArrayOptions): VertexArrayObject
}

external interface `T$0` {
    var size: Number
}

external interface Buffer : Resource {
    var stats: `T$0`
    @nativeInvoke
    operator fun invoke(data: Array<Number>): Buffer
    @nativeInvoke
    operator fun invoke(data: Array<Array<Number>>): Buffer
    @nativeInvoke
    operator fun invoke(data: Uint8Array): Buffer
    @nativeInvoke
    operator fun invoke(data: Int8Array): Buffer
    @nativeInvoke
    operator fun invoke(data: Uint16Array): Buffer
    @nativeInvoke
    operator fun invoke(data: Int16Array): Buffer
    @nativeInvoke
    operator fun invoke(data: Uint32Array): Buffer
    @nativeInvoke
    operator fun invoke(data: Int32Array): Buffer
    @nativeInvoke
    operator fun invoke(data: Float32Array): Buffer
    @nativeInvoke
    operator fun invoke(options: BufferOptions): Buffer
    fun subdata(data: Array<Number>, offset: Number = definedExternally): Buffer
    fun subdata(data: Array<Number>): Buffer
    fun subdata(data: Array<Array<Number>>, offset: Number = definedExternally): Buffer
    fun subdata(data: Array<Array<Number>>): Buffer
    fun subdata(data: Uint8Array, offset: Number = definedExternally): Buffer
    fun subdata(data: Uint8Array): Buffer
    fun subdata(data: Int8Array, offset: Number = definedExternally): Buffer
    fun subdata(data: Int8Array): Buffer
    fun subdata(data: Uint16Array, offset: Number = definedExternally): Buffer
    fun subdata(data: Uint16Array): Buffer
    fun subdata(data: Int16Array, offset: Number = definedExternally): Buffer
    fun subdata(data: Int16Array): Buffer
    fun subdata(data: Uint32Array, offset: Number = definedExternally): Buffer
    fun subdata(data: Uint32Array): Buffer
    fun subdata(data: Int32Array, offset: Number = definedExternally): Buffer
    fun subdata(data: Int32Array): Buffer
    fun subdata(data: Float32Array, offset: Number = definedExternally): Buffer
    fun subdata(data: Float32Array): Buffer
    fun subdata(options: BufferOptions, offset: Number = definedExternally): Buffer
    fun subdata(options: BufferOptions): Buffer
}

external interface BufferOptions {
    var data: dynamic /* Array<Number>? | Array<Array<Number>>? | Uint8Array? | Int8Array? | Uint16Array? | Int16Array? | Uint32Array? | Int32Array? | Float32Array? */
        get() = definedExternally
        set(value) = definedExternally
    var length: Number?
        get() = definedExternally
        set(value) = definedExternally
    var usage: String? /* "static" | "dynamic" | "stream" */
        get() = definedExternally
        set(value) = definedExternally
    var type: String? /* "uint8" | "int8" | "uint16" | "int16" | "uint32" | "int32" | "float32" | "float" */
        get() = definedExternally
        set(value) = definedExternally
}

external interface Elements : Resource {
    @nativeInvoke
    operator fun invoke(data: Array<Number>): Elements
    @nativeInvoke
    operator fun invoke(data: Array<Array<Number>>): Elements
    @nativeInvoke
    operator fun invoke(data: Uint8Array): Elements
    @nativeInvoke
    operator fun invoke(data: Uint16Array): Elements
    @nativeInvoke
    operator fun invoke(data: Uint32Array): Elements
    @nativeInvoke
    operator fun invoke(options: ElementsOptions): Elements
    fun subdata(data: Array<Number>, offset: Number = definedExternally): Elements
    fun subdata(data: Array<Number>): Elements
    fun subdata(data: Array<Array<Number>>, offset: Number = definedExternally): Elements
    fun subdata(data: Array<Array<Number>>): Elements
    fun subdata(data: Uint8Array, offset: Number = definedExternally): Elements
    fun subdata(data: Uint8Array): Elements
    fun subdata(data: Uint16Array, offset: Number = definedExternally): Elements
    fun subdata(data: Uint16Array): Elements
    fun subdata(data: Uint32Array, offset: Number = definedExternally): Elements
    fun subdata(data: Uint32Array): Elements
    fun subdata(options: ElementsOptions, offset: Number = definedExternally): Elements
    fun subdata(options: ElementsOptions): Elements
}

external interface ElementsOptions {
    var data: dynamic /* Array<Number>? | Array<Array<Number>>? | Uint8Array? | Uint16Array? | Uint32Array? */
        get() = definedExternally
        set(value) = definedExternally
    var usage: String? /* "static" | "dynamic" | "stream" */
        get() = definedExternally
        set(value) = definedExternally
    var length: Number?
        get() = definedExternally
        set(value) = definedExternally
    var primitive: String? /* "points" | "lines" | "line strip" | "line loop" | "triangles" | "triangle strip" | "triangle fan" */
        get() = definedExternally
        set(value) = definedExternally
    var type: String? /* "uint8" | "uint16" | "uint32" */
        get() = definedExternally
        set(value) = definedExternally
    var count: Number?
        get() = definedExternally
        set(value) = definedExternally
}

external interface VertexArrayOptions {
    var attributes: Array<dynamic /* ConstantAttribute | AttributeConfig | io.kinference.externals.Buffer | Array<Number> | Array<Array<Number>> | Uint8Array | Int8Array | Uint16Array | Int16Array | Uint32Array | Int32Array | Float32Array */>
    var elements: dynamic /* io.kinference.externals.Elements? | io.kinference.externals.ElementsOptions? | Array<Number>? | Array<Array<Number>>? | Uint8Array? | Uint16Array? | Uint32Array? */
        get() = definedExternally
        set(value) = definedExternally
    var count: Number?
        get() = definedExternally
        set(value) = definedExternally
    var offset: Number?
        get() = definedExternally
        set(value) = definedExternally
    var primitive: String? /* "points" | "lines" | "line strip" | "line loop" | "triangles" | "triangle strip" | "triangle fan" */
        get() = definedExternally
        set(value) = definedExternally
    var instances: Number?
        get() = definedExternally
        set(value) = definedExternally
}

external interface Texture : Resource {
    var stats: `T$0`
    var width: Number
    var height: Number
    var format: String /* "alpha" | "luminance" | "luminance alpha" | "rgb" | "rgba" | "rgba4" | "rgb5 a1" | "rgb565" | "srgb" | "srgba" | "depth" | "depth stencil" | "rgb s3tc dxt1" | "rgba s3tc dxt1" | "rgba s3tc dxt3" | "rgba s3tc dxt5" | "rgb atc" | "rgba atc explicit alpha" | "rgba atc interpolated alpha" | "rgb pvrtc 4bppv1" | "rgb pvrtc 2bppv1" | "rgba pvrtc 4bppv1" | "rgba pvrtc 2bppv1" | "rgb etc1" */
    var type: String /* "uint8" | "uint16" | "uint32" | "float" | "float32" | "half float" | "float16" */
    var mag: String /* "nearest" | "linear" */
    var min: String /* "nearest" | "linear" | "linear mipmap linear" | "mipmap" | "nearest mipmap linear" | "linear mipmap nearest" | "nearest mipmap nearest" */
    var wrapS: String /* "repeat" | "clamp" | "mirror" */
    var wrapT: String /* "repeat" | "clamp" | "mirror" */
}

external interface Texture2D : Texture {
    @nativeInvoke
    operator fun invoke(): Texture2D
    @nativeInvoke
    operator fun invoke(radius: Number): Texture2D
    @nativeInvoke
    operator fun invoke(width: Number, height: Number): Texture2D
    @nativeInvoke
    operator fun invoke(data: Array<Number>): Texture2D
    @nativeInvoke
    operator fun invoke(data: Array<Array<Number>>): Texture2D
    @nativeInvoke
    operator fun invoke(data: Array<Array<Array<Number>>>): Texture2D
    @nativeInvoke
    operator fun invoke(data: ArrayBufferView): Texture2D
    @nativeInvoke
    operator fun invoke(data: NDArrayLike): Texture2D
    @nativeInvoke
    operator fun invoke(data: HTMLImageElement): Texture2D
    @nativeInvoke
    operator fun invoke(data: HTMLVideoElement): Texture2D
    @nativeInvoke
    operator fun invoke(data: HTMLCanvasElement): Texture2D
    @nativeInvoke
    operator fun invoke(data: CanvasRenderingContext2D): Texture2D
    @nativeInvoke
    operator fun invoke(options: Texture2DOptions): Texture2D
    fun subimage(data: Array<Number>, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): Texture2D
    fun subimage(data: Array<Number>): Texture2D
    fun subimage(data: Array<Number>, x: Number = definedExternally): Texture2D
    fun subimage(data: Array<Number>, x: Number = definedExternally, y: Number = definedExternally): Texture2D
    fun subimage(data: Array<Array<Number>>, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): Texture2D
    fun subimage(data: Array<Array<Number>>): Texture2D
    fun subimage(data: Array<Array<Number>>, x: Number = definedExternally): Texture2D
    fun subimage(data: Array<Array<Number>>, x: Number = definedExternally, y: Number = definedExternally): Texture2D
    fun subimage(data: Array<Array<Array<Number>>>, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): Texture2D
    fun subimage(data: Array<Array<Array<Number>>>): Texture2D
    fun subimage(data: Array<Array<Array<Number>>>, x: Number = definedExternally): Texture2D
    fun subimage(data: Array<Array<Array<Number>>>, x: Number = definedExternally, y: Number = definedExternally): Texture2D
    fun subimage(data: ArrayBufferView, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): Texture2D
    fun subimage(data: ArrayBufferView): Texture2D
    fun subimage(data: ArrayBufferView, x: Number = definedExternally): Texture2D
    fun subimage(data: ArrayBufferView, x: Number = definedExternally, y: Number = definedExternally): Texture2D
    fun subimage(data: NDArrayLike, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): Texture2D
    fun subimage(data: NDArrayLike): Texture2D
    fun subimage(data: NDArrayLike, x: Number = definedExternally): Texture2D
    fun subimage(data: NDArrayLike, x: Number = definedExternally, y: Number = definedExternally): Texture2D
    fun subimage(data: HTMLImageElement, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): Texture2D
    fun subimage(data: HTMLImageElement): Texture2D
    fun subimage(data: HTMLImageElement, x: Number = definedExternally): Texture2D
    fun subimage(data: HTMLImageElement, x: Number = definedExternally, y: Number = definedExternally): Texture2D
    fun subimage(data: HTMLVideoElement, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): Texture2D
    fun subimage(data: HTMLVideoElement): Texture2D
    fun subimage(data: HTMLVideoElement, x: Number = definedExternally): Texture2D
    fun subimage(data: HTMLVideoElement, x: Number = definedExternally, y: Number = definedExternally): Texture2D
    fun subimage(data: HTMLCanvasElement, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): Texture2D
    fun subimage(data: HTMLCanvasElement): Texture2D
    fun subimage(data: HTMLCanvasElement, x: Number = definedExternally): Texture2D
    fun subimage(data: HTMLCanvasElement, x: Number = definedExternally, y: Number = definedExternally): Texture2D
    fun subimage(data: CanvasRenderingContext2D, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): Texture2D
    fun subimage(data: CanvasRenderingContext2D): Texture2D
    fun subimage(data: CanvasRenderingContext2D, x: Number = definedExternally): Texture2D
    fun subimage(data: CanvasRenderingContext2D, x: Number = definedExternally, y: Number = definedExternally): Texture2D
    fun subimage(options: Texture2DOptions, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): Texture2D
    fun subimage(options: Texture2DOptions): Texture2D
    fun subimage(options: Texture2DOptions, x: Number = definedExternally): Texture2D
    fun subimage(options: Texture2DOptions, x: Number = definedExternally, y: Number = definedExternally): Texture2D
    fun resize(radius: Number): Texture2D
    fun resize(width: Number, height: Number): Texture2D
}

external interface Texture2DOptions {
    var shape: dynamic /* JsTuple<Number, Number> | JsTuple<Number, Number, Number> */
        get() = definedExternally
        set(value) = definedExternally
    var radius: Number?
        get() = definedExternally
        set(value) = definedExternally
    var width: Number?
        get() = definedExternally
        set(value) = definedExternally
    var height: Number?
        get() = definedExternally
        set(value) = definedExternally
    var channels: Number? /* 1 | 2 | 3 | 4 */
        get() = definedExternally
        set(value) = definedExternally
    var data: dynamic /* Array<Number>? | Array<Array<Number>>? | Array<Array<Array<Number>>>? | ArrayBufferView? | io.kinference.externals.NDArrayLike? | HTMLImageElement? | HTMLVideoElement? | HTMLCanvasElement? | CanvasRenderingContext2D? */
        get() = definedExternally
        set(value) = definedExternally
    var copy: Boolean?
        get() = definedExternally
        set(value) = definedExternally
    var mag: String? /* "nearest" | "linear" */
        get() = definedExternally
        set(value) = definedExternally
    var min: String? /* "nearest" | "linear" | "linear mipmap linear" | "mipmap" | "nearest mipmap linear" | "linear mipmap nearest" | "nearest mipmap nearest" */
        get() = definedExternally
        set(value) = definedExternally
    var wrap: dynamic /* "repeat" | "clamp" | "mirror" | JsTuple<String, String> */
        get() = definedExternally
        set(value) = definedExternally
    var wrapS: String? /* "repeat" | "clamp" | "mirror" */
        get() = definedExternally
        set(value) = definedExternally
    var wrapT: String? /* "repeat" | "clamp" | "mirror" */
        get() = definedExternally
        set(value) = definedExternally
    var aniso: Number?
        get() = definedExternally
        set(value) = definedExternally
    var format: String? /* "alpha" | "luminance" | "luminance alpha" | "rgb" | "rgba" | "rgba4" | "rgb5 a1" | "rgb565" | "srgb" | "srgba" | "depth" | "depth stencil" | "rgb s3tc dxt1" | "rgba s3tc dxt1" | "rgba s3tc dxt3" | "rgba s3tc dxt5" | "rgb atc" | "rgba atc explicit alpha" | "rgba atc interpolated alpha" | "rgb pvrtc 4bppv1" | "rgb pvrtc 2bppv1" | "rgba pvrtc 4bppv1" | "rgba pvrtc 2bppv1" | "rgb etc1" */
        get() = definedExternally
        set(value) = definedExternally
    var type: String? /* "uint8" | "uint16" | "uint32" | "float" | "float32" | "half float" | "float16" */
        get() = definedExternally
        set(value) = definedExternally
    var mipmap: dynamic /* Boolean? | "don't care" | "dont care" | "nice" | "fast" | Array<Array<Number>>? */
        get() = definedExternally
        set(value) = definedExternally
    var flipY: Boolean?
        get() = definedExternally
        set(value) = definedExternally
    var alignment: Number? /* 1 | 2 | 4 | 8 */
        get() = definedExternally
        set(value) = definedExternally
    var premultiplyAlpha: Boolean?
        get() = definedExternally
        set(value) = definedExternally
    var colorSpace: String? /* "none" | "browser" */
        get() = definedExternally
        set(value) = definedExternally
}

external interface NDArrayLike {
    var shape: Array<Number>
    var stride: Array<Number>
    var offset: Number
    var data: dynamic /* Array<Number> | ArrayBufferView */
        get() = definedExternally
        set(value) = definedExternally
}

external interface TextureCube : Texture {
    @nativeInvoke
    operator fun invoke(): TextureCube
    @nativeInvoke
    operator fun invoke(radius: Number): TextureCube
    @nativeInvoke
    operator fun invoke(posXData: Array<Number>, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    @nativeInvoke
    operator fun invoke(posXData: Array<Array<Number>>, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    @nativeInvoke
    operator fun invoke(posXData: Array<Array<Array<Number>>>, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    @nativeInvoke
    operator fun invoke(posXData: ArrayBufferView, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    @nativeInvoke
    operator fun invoke(posXData: NDArrayLike, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    @nativeInvoke
    operator fun invoke(posXData: HTMLImageElement, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    @nativeInvoke
    operator fun invoke(posXData: HTMLVideoElement, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    @nativeInvoke
    operator fun invoke(posXData: HTMLCanvasElement, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    @nativeInvoke
    operator fun invoke(posXData: CanvasRenderingContext2D, negXData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negYData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, posZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */, negZData: Any /* Array<Number> | Array<Array<Number>> | Array<Array<Array<Number>>> | ArrayBufferView | io.kinference.externals.NDArrayLike | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | CanvasRenderingContext2D */): TextureCube
    @nativeInvoke
    operator fun invoke(posXOptions: Texture2DOptions, negXOptions: Texture2DOptions, posYOptions: Texture2DOptions, negYOptions: Texture2DOptions, posZOptions: Texture2DOptions, negZOptions: Texture2DOptions): TextureCube
    @nativeInvoke
    operator fun invoke(options: TextureCubeOptions): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: Array<Number>, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: Array<Number>): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: Array<Number>, x: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: Array<Number>, x: Number = definedExternally, y: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: Array<Array<Number>>, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: Array<Array<Number>>): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: Array<Array<Number>>, x: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: Array<Array<Number>>, x: Number = definedExternally, y: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: Array<Array<Array<Number>>>, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: Array<Array<Array<Number>>>): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: Array<Array<Array<Number>>>, x: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: Array<Array<Array<Number>>>, x: Number = definedExternally, y: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: ArrayBufferView, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: ArrayBufferView): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: ArrayBufferView, x: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: ArrayBufferView, x: Number = definedExternally, y: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: NDArrayLike, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: NDArrayLike): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: NDArrayLike, x: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: NDArrayLike, x: Number = definedExternally, y: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: HTMLImageElement, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: HTMLImageElement): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: HTMLImageElement, x: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: HTMLImageElement, x: Number = definedExternally, y: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: HTMLVideoElement, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: HTMLVideoElement): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: HTMLVideoElement, x: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: HTMLVideoElement, x: Number = definedExternally, y: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: HTMLCanvasElement, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: HTMLCanvasElement): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: HTMLCanvasElement, x: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: HTMLCanvasElement, x: Number = definedExternally, y: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: CanvasRenderingContext2D, x: Number = definedExternally, y: Number = definedExternally, level: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: CanvasRenderingContext2D): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: CanvasRenderingContext2D, x: Number = definedExternally): TextureCube
    fun subimage(face: Number /* 0 | 1 | 2 | 3 | 4 | 5 */, data: CanvasRenderingContext2D, x: Number = definedExternally, y: Number = definedExternally): TextureCube
    fun resize(radius: Number): TextureCube
}

external interface TextureCubeOptions : Texture2DOptions {
    var faces: dynamic /* JsTuple<dynamic, dynamic, dynamic, dynamic, dynamic, dynamic> */
        get() = definedExternally
        set(value) = definedExternally
}

external interface Renderbuffer : Resource {
    var stats: `T$0`
    var width: Number
    var height: Number
    var format: Number
    @nativeInvoke
    operator fun invoke(): Renderbuffer
    @nativeInvoke
    operator fun invoke(radius: Number): Renderbuffer
    @nativeInvoke
    operator fun invoke(width: Number, height: Number): Renderbuffer
    @nativeInvoke
    operator fun invoke(options: RenderbufferOptions): Renderbuffer
    fun resize(radius: Number): Renderbuffer
    fun resize(width: Number, height: Number): Renderbuffer
}

external interface RenderbufferOptions {
    var shape: dynamic /* JsTuple<Number, Number> */
        get() = definedExternally
        set(value) = definedExternally
    var radius: Number?
        get() = definedExternally
        set(value) = definedExternally
    var width: Number?
        get() = definedExternally
        set(value) = definedExternally
    var height: Number?
        get() = definedExternally
        set(value) = definedExternally
    var format: String? /* "rgba4" | "rgb565" | "rgb5 a1" | "rgb16f" | "rgba16f" | "rgba32f" | "srgba" | "depth" | "stencil" | "depth stencil" */
        get() = definedExternally
        set(value) = definedExternally
}

external interface Framebuffer2D : Resource {
    @nativeInvoke
    operator fun invoke(): Framebuffer2D
    @nativeInvoke
    operator fun invoke(radius: Number): Framebuffer2D
    @nativeInvoke
    operator fun invoke(width: Number, height: Number): Framebuffer2D
    @nativeInvoke
    operator fun invoke(options: FramebufferOptions): Framebuffer2D
    fun <Context : DefaultContext, Props : Any> use(body: CommandBodyFn<Context, Props>)
    fun resize(radius: Number): Framebuffer2D
    fun resize(width: Number, height: Number): Framebuffer2D
}

external interface FramebufferOptions {
    var shape: dynamic /* JsTuple<Number, Number> */
        get() = definedExternally
        set(value) = definedExternally
    var radius: Number?
        get() = definedExternally
        set(value) = definedExternally
    var width: Number?
        get() = definedExternally
        set(value) = definedExternally
    var height: Number?
        get() = definedExternally
        set(value) = definedExternally
    var color: dynamic /* io.kinference.externals.Texture2D? | io.kinference.externals.Renderbuffer? */
        get() = definedExternally
        set(value) = definedExternally
    var colors: Array<dynamic /* io.kinference.externals.Texture2D | io.kinference.externals.Renderbuffer */>?
        get() = definedExternally
        set(value) = definedExternally
    var colorFormat: String? /* "rgba" | "rgba4" | "rgb565" | "rgb5 a1" | "rgb16f" | "rgba16f" | "rgba32f" | "srgba" */
        get() = definedExternally
        set(value) = definedExternally
    var colorType: String? /* "uint8" | "half float" | "float" */
        get() = definedExternally
        set(value) = definedExternally
    var colorCount: Number?
        get() = definedExternally
        set(value) = definedExternally
    var depth: dynamic /* Boolean? | io.kinference.externals.Texture2D? | io.kinference.externals.Renderbuffer? */
        get() = definedExternally
        set(value) = definedExternally
    var stencil: dynamic /* Boolean? | io.kinference.externals.Texture2D? | io.kinference.externals.Renderbuffer? */
        get() = definedExternally
        set(value) = definedExternally
    var depthStencil: dynamic /* Boolean? | io.kinference.externals.Texture2D? | io.kinference.externals.Renderbuffer? */
        get() = definedExternally
        set(value) = definedExternally
    var depthTexture: Boolean?
        get() = definedExternally
        set(value) = definedExternally
}

external interface FramebufferCube : Resource {
    @nativeInvoke
    operator fun invoke(): FramebufferCube
    @nativeInvoke
    operator fun invoke(radius: Number): FramebufferCube
    @nativeInvoke
    operator fun invoke(options: FramebufferCubeOptions): FramebufferCube
    fun resize(radius: Number): FramebufferCube
    var faces: dynamic /* JsTuple<dynamic, dynamic, dynamic, dynamic, dynamic> */
        get() = definedExternally
        set(value) = definedExternally
}

external interface FramebufferCubeOptions {
    var shape: dynamic /* JsTuple<Number, Number> */
        get() = definedExternally
        set(value) = definedExternally
    var radius: Number?
        get() = definedExternally
        set(value) = definedExternally
    var width: Number?
        get() = definedExternally
        set(value) = definedExternally
    var height: Number?
        get() = definedExternally
        set(value) = definedExternally
    var color: TextureCube?
        get() = definedExternally
        set(value) = definedExternally
    var colors: Array<TextureCube>?
        get() = definedExternally
        set(value) = definedExternally
    var colorFormat: String? /* "rgba" */
        get() = definedExternally
        set(value) = definedExternally
    var colorType: String? /* "uint8" | "half float" | "float" */
        get() = definedExternally
        set(value) = definedExternally
    var colorCount: Number?
        get() = definedExternally
        set(value) = definedExternally
    var depth: dynamic /* Boolean? | io.kinference.externals.TextureCube? */
        get() = definedExternally
        set(value) = definedExternally
    var stencil: dynamic /* Boolean? | io.kinference.externals.TextureCube? */
        get() = definedExternally
        set(value) = definedExternally
    var depthStencil: dynamic /* Boolean? | io.kinference.externals.TextureCube? */
        get() = definedExternally
        set(value) = definedExternally
}

external interface Limits {
    var colorBits: dynamic /* JsTuple<Number, Number, Number, Number> */
        get() = definedExternally
        set(value) = definedExternally
    var depthBits: Number
    var stencilBits: Number
    var subpixelBits: Number
    var extensions: Array<String>
    var maxAnisotropic: Number
    var maxDrawbuffers: Number
    var maxColorAttachments: Number
    var pointSizeDims: Float32Array
    var lineWidthDims: Float32Array
    var maxViewportDims: Int32Array
    var maxCombinedTextureUnits: Number
    var maxCubeMapSize: Number
    var maxRenderbufferSize: Number
    var maxTextureUnits: Number
    var maxTextureSize: Number
    var maxAttributes: Number
    var maxVertexUniforms: Number
    var maxVertexTextureUnits: Number
    var maxVaryingVectors: Number
    var maxFragmentUniforms: Number
    var glsl: String
    var renderer: String
    var vendor: String
    var version: String
    var textureFormats: Array<String /* "alpha" | "luminance" | "luminance alpha" | "rgb" | "rgba" | "rgba4" | "rgb5 a1" | "rgb565" | "srgb" | "srgba" | "depth" | "depth stencil" | "rgb s3tc dxt1" | "rgba s3tc dxt1" | "rgba s3tc dxt3" | "rgba s3tc dxt5" | "rgb atc" | "rgba atc explicit alpha" | "rgba atc interpolated alpha" | "rgb pvrtc 4bppv1" | "rgb pvrtc 2bppv1" | "rgba pvrtc 4bppv1" | "rgba pvrtc 2bppv1" | "rgb etc1" */>
}

external interface Stats {
    var bufferCount: Number
    var elementsCount: Number
    var framebufferCount: Number
    var shaderCount: Number
    var textureCount: Number
    var cubeCount: Number
    var renderbufferCount: Number
    var maxTextureUnits: Number
    var vaoCount: Number
    var getTotalTextureSize: (() -> Number)?
        get() = definedExternally
        set(value) = definedExternally
    var getTotalBufferSize: (() -> Number)?
        get() = definedExternally
        set(value) = definedExternally
    var getTotalRenderbufferSize: (() -> Number)?
        get() = definedExternally
        set(value) = definedExternally
    var getMaxUniformsCount: (() -> Number)?
        get() = definedExternally
        set(value) = definedExternally
    var getMaxAttributesCount: (() -> Number)?
        get() = definedExternally
        set(value) = definedExternally
}

external interface CommandStats {
    var count: Number
    var cpuTime: Number
    var gpuTime: Number
}
