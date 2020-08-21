plugins {
    `maven-publish`
    kotlin("jvm")
}

group = "org.jetbrains.research.kotlin.inference"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            groupId = "org.jetbrains.research.kotlin.inference"
            artifactId = "annotations"
            version = "0.1.0"

            from(components["kotlin"])
        }
    }

    repositories {
        mavenLocal()
    }
}
