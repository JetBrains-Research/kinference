package io.kinference.utils.wgpu.generation.generators

import com.squareup.kotlinpoet.*
import com.squareup.kotlinpoet.ParameterizedTypeName.Companion.parameterizedBy
import io.kinference.utils.wgpu.generation.generators.base.ClassGenerator
import io.kinference.utils.wgpu.generation.models.*
import io.kinference.utils.wgpu.generation.utils.TypeMapper
import io.kinference.utils.wgpu.internal.MemoryMode
import io.kinference.utils.wgpu.internal.WgpuRuntime
import jnr.ffi.*
import kotlin.reflect.KClass

fun generateStructDeclaration(
    packageName: String,
    structDeclaration: StructDeclaration,
    typeMapper: TypeMapper,
): TypeSpec = StructDeclarationGenerator(packageName, structDeclaration, typeMapper).generate()

class StructDeclarationGenerator(
    private val packageName: String,
    private val structDeclaration: StructDeclaration,
    private val typeMapper: TypeMapper
) : ClassGenerator(ClassName(packageName, structDeclaration.name)) {
    override fun generateImpl() {
        builder.apply {
            addModifiers(KModifier.OPEN)
            superclass(Struct::class)
            addSuperclassConstructorParameter("runtime")

            if (structDeclaration.fields.any {
                    val type = typeMapper.getType(it.type)
                    type is CStringType || type is PointerType || type is StructPointerType
            }) {
                addSuperclassConstructorParameter("Alignment(8)")
            }

            primaryConstructor(
                FunSpec.constructorBuilder()
                    .addParameter("runtime", Runtime::class)
                    .build()
            )
            addFunction(
                FunSpec.constructorBuilder()
                    .addModifiers(KModifier.PROTECTED)
                    .callThisConstructor(CodeBlock.of("%T.runtime", WgpuRuntime::class))
                    .build()
            )
            addFunction(
                FunSpec.constructorBuilder()
                    .addParameter("memoryMode", MemoryMode::class)
                    .callThisConstructor()
                    .addCode("""
                            |if (memoryMode == %T.%L) {
                            |    %M()
                            |}
                        """.trimMargin(),
                        MemoryMode::class,
                        MemoryMode.Direct,
                        MemberName("io.kinference.utils.wgpu.internal", "useDirectMemory", isExtension = true),
                    )
                    .build()
            )

            addType(generateCompanionObject())

            structDeclaration.fields.forEach { field ->
                addProperty(generateField(field))
            }
        }
    }

    private fun generateCompanionObject(): TypeSpec =
        TypeSpec.companionObjectBuilder()
            .apply {
                addFunction(
                    FunSpec.builder("allocate")
                        .addStatement("return ${structDeclaration.name}(%T.%L)", MemoryMode::class, MemoryMode.Heap)
                        .build()
                )
                addFunction(
                    FunSpec.builder("allocateDirect")
                        .addStatement("return ${structDeclaration.name}(%T.%L)", MemoryMode::class, MemoryMode.Direct)
                        .build()
                )
            }
            .build()

    private fun generateField(field: StructField): PropertySpec =
        when (val fieldType = typeMapper.getType(field.type)) {
            BooleanType -> generatePrimitiveField(field.name, Struct.Boolean::class, Boolean::class)
            Unsigned16Type -> generatePrimitiveField(field.name, Struct.Unsigned16::class, Int::class)
            Unsigned32Type -> generatePrimitiveField(field.name, Struct.Unsigned32::class, Long::class)
            Unsigned64Type -> generatePrimitiveField(field.name, Struct.Unsigned64::class, Long::class)
            Signed16Type -> generatePrimitiveField(field.name, Struct.Signed16::class, Short::class)
            Signed32Type -> generatePrimitiveField(field.name, Struct.Signed32::class, Int::class)
            Signed64Type -> generatePrimitiveField(field.name, Struct.Signed64::class, Long::class)
            FloatType -> generatePrimitiveField(field.name, Struct.Float::class, Float::class)
            DoubleType -> generatePrimitiveField(field.name, Struct.Double::class, Double::class)
            PointerType, CStringType -> generatePrimitiveField(
                field.name, Struct.Pointer::class, Pointer::class,
                "${field.type.name}${if (field.type.pointer) " *" else ""}"
            )
            is StructType -> generateStructField(field.name, ClassName(packageName, fieldType.name))
            is StructPointerType -> generateStructPointerField(field.name, ClassName(packageName, fieldType.name))
            is EnumType -> generateEnumField(field.name, ClassName(packageName, fieldType.name))
        }

    private fun generatePrimitiveField(
        fieldName: String,
        jnrType: KClass<*>,
        jvmType: KClass<*>,
        comment: String? = null
    ): PropertySpec {
        generateFieldView(fieldName, jvmType.asTypeName(), comment)
        return privateFieldBuilder(fieldName, jnrType.asTypeName()).apply {
            initializer("${jnrType.simpleName}()")
        }.build()
    }

    private fun generateStructField(fieldName: String, jnrType: TypeName, comment: String? = null): PropertySpec =
        publicFieldBuilder(fieldName, jnrType, comment).apply {
            initializer("inner(%T.allocate())", jnrType)
        }.build()

    private fun generateEnumField(fieldName: String, jnrType: TypeName, comment: String? = null): PropertySpec {
        generateFieldView(fieldName, jnrType, comment)
        return publicFieldBuilder("_$fieldName", Struct.Enum::class.asClassName().parameterizedBy(jnrType)).apply {
            initializer("Enum(%T::class.java)", jnrType)
        }.build()
    }

    private fun generateStructPointerField(fieldName: String, jnrType: TypeName, comment: String? = null): PropertySpec =
        publicFieldBuilder(fieldName, Struct.StructRef::class.asClassName().parameterizedBy(jnrType), comment).apply {
            initializer("StructRef(%T::class.java)", jnrType)
        }.build()

    private fun generateFieldView(fieldName: String, jvmType: TypeName, comment: String?) {
        PropertySpec.builder(fieldName, jvmType).apply {
            mutable()
            setter(FunSpec.setterBuilder()
                .addParameter("value", jvmType)
                .addStatement("return _$fieldName.set(%N)", "value")
                .build()
            )
            getter(FunSpec.getterBuilder()
                .addStatement("return _$fieldName.get()")
                .build()
            )
            comment?.let {
                addKdoc(it)
            }
        }.build().apply { builder.addProperty(this) }
    }

    private fun publicFieldBuilder(fieldName: String, jnrType: TypeName, comment: String? = null): PropertySpec.Builder =
        PropertySpec.builder(fieldName, jnrType).apply {
            addAnnotation(JvmField::class)
            comment?.let {
                addKdoc(it)
            }
        }

    private fun privateFieldBuilder(fieldName: String, jnrType: TypeName): PropertySpec.Builder =
        PropertySpec.builder("_$fieldName", jnrType).apply {
            addModifiers(KModifier.PRIVATE)
        }
}
