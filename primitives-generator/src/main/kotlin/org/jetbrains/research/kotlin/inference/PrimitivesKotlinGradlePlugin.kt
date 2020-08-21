package org.jetbrains.research.kotlin.inference

import org.gradle.api.DefaultTask
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.tasks.TaskAction
import org.jetbrains.kotlin.com.intellij.psi.PsiElement
import org.jetbrains.kotlin.com.intellij.psi.impl.source.tree.LeafPsiElement
import org.jetbrains.kotlin.lexer.KtTokens.IDENTIFIER
import org.jetbrains.kotlin.psi.*
import org.jetbrains.kotlin.resolve.BindingContext
import org.jetbrains.kotlin.resolve.descriptorUtil.fqNameSafe
import org.jetbrains.research.kotlin.inference.annotations.*
import java.io.File
import kotlin.reflect.KClass

abstract class KtDefaultVisitor : KtVisitorVoid() {
    protected open fun shouldVisitElement(element: PsiElement) = true

    override fun visitElement(element: PsiElement) {
        if (!shouldVisitElement(element)) return

        if (element is LeafPsiElement) visitLeafElement(element)
        else element.acceptChildren(this)
    }

    open fun visitLeafElement(element: LeafPsiElement) {}
}

val KtNamedDeclaration.qualifiedName
    get() = fqName?.asString() ?: error("FqName not found")

fun KtAnnotationEntry.getDescriptor(context: BindingContext) = context[BindingContext.ANNOTATION, this]!!.forced()

inline fun <reified T : Annotation> KtAnnotationEntry.isAnnotation(context: BindingContext): Boolean {
    return getDescriptor(context).fqName?.asString() == T::class.qualifiedName
}

inline fun <reified T : Annotation> KtAnnotated.isAnnotatedWith(context: BindingContext): Boolean {
    return annotationEntries.any { it.isAnnotation<T>(context) }
}

class Primitive<Type : Any, ArrayType : Any>(val dataType: DataType, val type: KClass<Type>, val arrayType: KClass<ArrayType>) {
    companion object {
        inline fun <reified Type : Any, reified ArrayType : Any> create(dataType: DataType): Primitive<Type, ArrayType> {
            return Primitive(dataType, Type::class, ArrayType::class)
        }
    }

    val typeName = type.simpleName
    val arrayTypeName = arrayType.simpleName
}

@ExperimentalUnsignedTypes
open class GenerateSources : DefaultTask() {
    @TaskAction
    fun act() {
        val sources = project.myKtSourceSet
        val classes = project.configurations.getByName("compileClasspath").files

        val manager = EnvironmentManager.create(classes)
        val ktFiles = ParseUtil.analyze(sources, manager)
        val context = ResolveUtil.analyze(ktFiles, manager).bindingContext

        val primitives = mapOf(
            Primitive.create<Byte, ByteArray>(DataType.BYTE) to StringBuilder(),
            Primitive.create<Short, ShortArray>(DataType.SHORT) to StringBuilder(),
            Primitive.create<Int, IntArray>(DataType.INT) to StringBuilder(),
            Primitive.create<Long, LongArray>(DataType.LONG) to StringBuilder(),

            Primitive.create<UByte, UByteArray>(DataType.UBYTE) to StringBuilder(),
            Primitive.create<UShort, UShortArray>(DataType.USHORT) to StringBuilder(),
            Primitive.create<UInt, UIntArray>(DataType.UINT) to StringBuilder(),
            Primitive.create<ULong, ULongArray>(DataType.ULONG) to StringBuilder(),

            Primitive.create<Float, FloatArray>(DataType.FLOAT) to StringBuilder(),
            Primitive.create<Double, DoubleArray>(DataType.DOUBLE) to StringBuilder()
        )

        for (file in ktFiles) {
            if (!file.isAnnotatedWith<GenerateWithPrimitives>(context)) continue

            val classes = HashSet<KtClass>()
            file.accept(object : KtDefaultVisitor() {
                override fun visitClass(klass: KtClass) {
                    if (klass.isAnnotatedWith<PrimitiveClass>(context)) {
                        classes.add(klass)
                    }
                }
            })

            for ((primitive, builder) in primitives) {
                val replacements = HashMap<String, String>().apply {
                    put(DataType::class.qualifiedName!! + ".${DataType.UNKNOWN.name}", primitive.dataType.name)

                    put(PrimitiveType::class.qualifiedName!! + ".toPrimitive", "to${primitive.typeName}")
                    put("org.jetbrains.research.kotlin.inference.annotations.toPrimitive", "to${primitive.typeName}")

                    put(PrimitiveType::class.qualifiedName!!, primitive.typeName!!)
                    put(PrimitiveType::class.qualifiedName!! + ".<init>", primitive.typeName)
                    put(PrimitiveType::class.qualifiedName!! + ".Companion", primitive.typeName)

                    put(PrimitiveArray::class.qualifiedName!!, primitive.arrayTypeName!!)
                    put(PrimitiveArray::class.qualifiedName!! + ".<init>", primitive.arrayTypeName)
                    put(PrimitiveArray::class.qualifiedName!! + ".Companion", primitive.arrayTypeName)

                    for (klass in classes) {
                        val replacement = klass.name!!.replace("Primitive", primitive.typeName)
                        put(klass.qualifiedName, replacement)
                        put(klass.qualifiedName + ".<init>", replacement)
                        put(klass.qualifiedName + ".Companion", replacement)
                    }
                }

                file.accept(object : KtDefaultVisitor() {
                    override fun visitAnnotationEntry(annotationEntry: KtAnnotationEntry) {
                        if (annotationEntry.isAnnotation<GenerateWithPrimitives>(context) ||
                            annotationEntry.isAnnotation<PrimitiveClass>(context)) return

                        super.visitAnnotationEntry(annotationEntry)
                    }

                    override fun visitSimpleNameExpression(expression: KtSimpleNameExpression) {
                        val reference = context[BindingContext.REFERENCE_TARGET, expression]
                        if (reference != null) {
                            val type = reference.forced().fqNameSafe.asString()
                            if (expression.text != "this" && type in replacements) {
                                builder.append(replacements[type])
                                return
                            }
                        }

                        super.visitSimpleNameExpression(expression)
                    }

                    override fun visitLeafElement(element: LeafPsiElement) {
                        if (element.elementType == IDENTIFIER) {
                            if (element.parent in classes) {
                                builder.append(replacements[(element.parent as KtClass).qualifiedName])
                                return
                            }
                        }

                        builder.append(element.text)
                    }
                })

                val path = (project.extensions.getByName("primitives") as PrimitivesPluginExtension).generationPath

                with(File("${project.projectDir}/$path/${file.packageFqName.asString().replace('.', '/')}/" + file.name.replace("Primitive", primitive.typeName!!))) {
                    parentFile.mkdirs()
                    createNewFile()
                    writeText(builder.toString())
                }

                builder.clear()
            }
        }
    }
}

open class PrimitivesPluginExtension {
    var generationPath: String = "src/main/kotlin-gen"
}

class PrimitivesKotlinGradlePlugin : Plugin<Project> {
    override fun apply(target: Project) {
        val ext = target.extensions.create("primitives", PrimitivesPluginExtension::class.java)

        val task = target.tasks.create("generateSources", GenerateSources::class.java)
        target.tasks.getByName("classes").dependsOn.add(task)
    }
}
