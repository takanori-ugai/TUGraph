import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar
import io.gitlab.arturbosch.detekt.Detekt
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jlleitschuh.gradle.ktlint.reporter.ReporterType

plugins {
    kotlin("jvm") version "2.3.0"
    kotlin("plugin.serialization") version "2.3.0"
    java
    id("com.gradleup.shadow") version "9.3.1"
    jacoco
    id("org.jetbrains.dokka") version "2.1.0"
    id("io.gitlab.arturbosch.detekt") version "1.23.8"
//    id("com.github.sherter.google-java-format") version "0.9"
//    kotlin("jupyter.api") version "0.10.1-8"
    id("com.github.jk1.dependency-license-report") version "3.0.1"
    id("com.github.spotbugs") version "6.4.8"
    id("com.diffplug.spotless") version "8.2.1"
    id("org.jlleitschuh.gradle.ktlint") version "14.0.1"
    application
}

group = "jp.live.ugai"
version = "1.0-SNAPSHOT"
val v = "0.36.0"

repositories {
    mavenCentral()
    maven {
        url = uri("https://oss.sonatype.org/content/repositories/snapshots/")
    }
}

dependencies {
    implementation("ai.djl:basicdataset:$v")
    implementation("ai.djl:api:$v")
//    implementation("ai.djl.mxnet:mxnet-engine:$v")
    implementation("ai.djl.pytorch:pytorch-engine:$v")
    implementation("ai.djl.pytorch:pytorch-model-zoo:$v")
    implementation("ai.djl.huggingface:tokenizers:$v")
//    runtimeOnly("ai.djl.pytorch:pytorch-native-cpu:2.4.0")
    //    implementation("ai.djl.pytorch:pytorch-native-cpu:2.4.0:linux-x86_64")
//    runtimeOnly("ai.djl.pytorch:pytorch-native-cu124:2.4.0:linux-x86_64")
    runtimeOnly("ai.djl.pytorch:pytorch-jni:2.7.1-0.36.0")
    runtimeOnly("ai.djl.pytorch:pytorch-native-cu124:2.5.1:linux-x86_64")
    implementation("org.slf4j:slf4j-simple:2.0.17")
//    implementation(kotlin("stdlib"))
    implementation("com.opencsv:opencsv:5.12.0")
    testImplementation("org.junit.jupiter:junit-jupiter-api:6.0.2")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:6.0.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:6.0.2")
}

tasks {
    compileKotlin {
        compilerOptions.jvmTarget.set(JvmTarget.JVM_17)
    }

    compileTestKotlin {
        compilerOptions.jvmTarget.set(JvmTarget.JVM_17)
    }

    compileJava {
        options.encoding = "UTF-8"
        sourceCompatibility = "17"
        targetCompatibility = "17"
    }

    compileTestJava {
        options.encoding = "UTF-8"
        sourceCompatibility = "17"
        targetCompatibility = "17"
    }

    test {
        useJUnitPlatform()
        finalizedBy(jacocoTestReport) // report is always generated after tests run
    }

    withType<Detekt>().configureEach {
        // Target version of the generated JVM bytecode. It is used for type resolution.
        jvmTarget = "17"
        reports {
            // observe findings in your browser with structure and code snippets
            html.required.set(true)
            // checkstyle like format mainly for integrations like Jenkins
            xml.required.set(true)
            // similar to the console output, contains issue signature to manually edit baseline files
            txt.required.set(true)
            // standardized SARIF format (https://sarifweb.azurewebsites.net/) to support integrations
            // with Github Code Scanning
            sarif.required.set(true)
        }
    }

    check {
        dependsOn("ktlintCheck")
    }

    jacocoTestReport {
        dependsOn(test) // tests are required to run before generating the report
    }

    withType<ShadowJar> {
        manifest {
            attributes["Main-Class"] = "com.fujitsu.labs.virtualhome.MainKt"
        }
    }

    val execute by registering(JavaExec::class) {
        group = "application"
        mainClass.set(
            if (project.hasProperty("mainClass")) {
                project.property("mainClass") as String
            } else {
                application.mainClass.get()
            },
        )
        classpath = sourceSets.main.get().runtimeClasspath
    }
}

ktlint {
    verbose.set(true)
    outputToConsole.set(true)
    coloredOutput.set(true)
    reporters {
        reporter(ReporterType.CHECKSTYLE)
        reporter(ReporterType.JSON)
        reporter(ReporterType.HTML)
    }
    filter {
        exclude("**/style-violations.kt")
    }
}

detekt {
    buildUponDefaultConfig = true // preconfigure defaults
    allRules = false // activate all available (even unstable) rules.
    // point to your custom config defining rules to run, overwriting default behavior
    config.from(files("$projectDir/config/detekt.yml"))
//    baseline = file("$projectDir/config/baseline.xml") // a way of suppressing issues before introducing detekt
}

spotbugs {
    ignoreFailures.set(true)
}

jacoco {
    toolVersion = "0.8.13"
//    reportsDirectory.set(layout.buildDirectory.dir("customJacocoReportDir"))
}

application {
    mainClass.set("jp.live.ugai.tugraph.Test6Kt")
}

spotless {
    java {
        target("src/*/java/**/*.java")
        targetExclude("src/jte-classes/**/*.java", "jte-classes/**/*.java")
        // Use the default importOrder configuration
        importOrder()
        removeUnusedImports()

        // Choose one of these formatters.
        googleJavaFormat("1.28.0") // has its own section below
        formatAnnotations() // fixes formatting of type annotations, see below
    }
}
