group = rootProject.group
version = rootProject.version

kotlin {
    jvm()

    sourceSets {
        jvmMain {
            dependencies {
                api(project(":serialization:serializer-protobuf"))
                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-core"))

                api(libs.wire.runtime)
            }
        }
    }
}
