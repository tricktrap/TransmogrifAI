/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.utils.kryo

import java.util.TreeMap

import com.esotericsoftware.kryo.{Kryo, Registration, Serializer}
import com.esotericsoftware.kryo.serializers.{DefaultArraySerializers, DefaultSerializers}
import com.twitter.chill.algebird.AlgebirdRegistrar
import com.twitter.chill.avro.AvroSerializer
import org.apache.log4j.{Level, Logger}
import org.apache.avro.generic.{GenericData, GenericRecord}
import org.apache.avro.specific.SpecificRecordBase
import org.apache.spark.serializer.KryoRegistrator

import scala.collection.mutable.{WrappedArray => MWrappedArray}
import scala.reflect._
import scala.util.{Failure, Success, Try}


class OpKryoRegistratorBase extends KryoRegistrator {

  @transient private lazy val logger: Logger = Logger.getLogger(this.getClass)

  protected def doAvroRegistration[T <: SpecificRecordBase : ClassTag](kryo: Kryo): Registration =
    kryo.register(classTag[T].runtimeClass, AvroSerializer.SpecificRecordBinarySerializer[T])

  protected def doClassRegistration(kryo: Kryo)(seqPC: Class[_]*): Unit =
    seqPC.foreach { pC =>
      Try(pC.getDeclaredConstructor().newInstance().asInstanceOf[Serializer[_]]) match {
        case Success(ser) =>
          Try(kryo.register(pC, ser))
            .foreach(_ => info(s"Registered class with default serializer: ${pC.getCanonicalName}"))
        case Failure(_) =>
          Try(kryo.register(pC))
            .foreach(_ => info(s"Registered class without default serializer: ${pC.getCanonicalName}"))
      }
      // also register arrays of that class
      val arrayType = java.lang.reflect.Array.newInstance(pC, 0).getClass
      kryo.register(arrayType)
      info(s"Registered class: ${arrayType.getCanonicalName}")
    }

  final override def registerClasses(kryo: Kryo): Unit = {
    doClassRegistration(kryo)({
      Seq(classOf[Class[_]],
        classOf[GenericData],
        classOf[GenericRecord],
        classOf[Object],
        classOf[java.math.BigDecimal],
        classOf[java.math.BigInteger],
        classOf[java.util.HashMap[_, _]],
        classOf[scala.collection.Seq[_]],
        classOf[scala.collection.immutable.Map[_, _]],
        classOf[scala.math.BigDecimal],
        classOf[scala.math.BigInt],
        scala.collection.immutable.HashMap.empty[Any, Any].getClass,
        scala.collection.immutable.Map.empty[Any, Any].getClass,
        scala.collection.immutable.Set.empty[Any].getClass) ++
        classOf[DefaultArraySerializers].getDeclaredClasses() ++
        classOf[DefaultSerializers].getDeclaredClasses() ++
        OpKryoClasses.ArraysOfPrimitives ++
        OpKryoClasses.WrappedArrays ++
        OpKryoClasses.SparkClasses
    }: _*)

    // Avro generic-data array deserialization fails - hence providing workaround
    kryo.register(
      classOf[GenericData.Array[_]], new GenericJavaCollectionSerializer(classOf[java.util.ArrayList[_]]))

    // A bunch of anonymous classes
    kryo.register(Class.forName("breeze.linalg.DenseVector$mcD$sp"))
    kryo.register(Class.forName("scala.Enumeration$Val"))
    kryo.register(Class.forName("scala.collection.immutable.MapLike$$anon$2"))
    kryo.register(Class.forName("scala.collection.immutable.MapLike$ImmutableDefaultKeySet"))
    kryo.register(Class.forName("scala.math.Ordering$$anon$4"))
    kryo.register(Class.forName("scala.reflect.ClassTag$$anon$1"))

    new AlgebirdRegistrar().apply(kryo)

    registerCustomClasses(kryo)
  }

  /**
   * Override this method to register custom types
   *
   * @param kryo
   */
  protected def registerCustomClasses(kryo: Kryo): Unit = ()

  private def info(msg: => String): Unit = if (logger.getLevel == Level.INFO) logger.info(msg) else ()

}

private[op] case object OpKryoClasses {

  lazy val ArraysOfPrimitives: Seq[Class[_]] = Seq(
    Class.forName("[Z") /* boolean[] */,
    Class.forName("[B") /* byte[] */,
    Class.forName("[C") /* char[] */,
    Class.forName("[D") /* double[] */,
    Class.forName("[F") /* float[] */,
    Class.forName("[I") /* int[] */,
    Class.forName("[J") /* long[] */,
    Class.forName("[S") /* short[] */
  )

  lazy val WrappedArrays: Seq[Class[_]] = Seq(
    MWrappedArray.make(Array[Boolean]()).getClass,
    MWrappedArray.make(Array[Byte]()).getClass,
    MWrappedArray.make(Array[Char]()).getClass,
    MWrappedArray.make(Array[Double]()).getClass,
    MWrappedArray.make(Array[Float]()).getClass,
    MWrappedArray.make(Array[Int]()).getClass,
    MWrappedArray.make(Array[Long]()).getClass,
    MWrappedArray.make(Array[Short]()).getClass,
    MWrappedArray.make(Array[String]()).getClass
  )

  lazy val SparkClasses: Seq[Class[_]] = Seq(
    classOf[org.apache.spark.internal.io.FileCommitProtocol.TaskCommitMessage],
    classOf[org.apache.spark.ml.classification.DecisionTreeClassificationModel],
    classOf[org.apache.spark.ml.linalg.Vector],
    classOf[org.apache.spark.ml.param.BooleanParam],
    classOf[org.apache.spark.ml.param.DoubleParam],
    classOf[org.apache.spark.ml.param.FloatParam],
    classOf[org.apache.spark.ml.param.IntParam],
    classOf[org.apache.spark.ml.param.LongParam],
    classOf[org.apache.spark.ml.regression.DecisionTreeRegressionModel],
    classOf[org.apache.spark.ml.tree.ContinuousSplit],
    classOf[org.apache.spark.ml.tree.Split],
    classOf[org.apache.spark.mllib.linalg.Vector],
    classOf[org.apache.spark.mllib.stat.MultivariateOnlineSummarizer],
    classOf[org.apache.spark.sql.Row],
    classOf[org.apache.spark.sql.catalyst.InternalRow],
    classOf[org.apache.spark.sql.catalyst.expressions.BoundReference],
    classOf[org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema],
    classOf[org.apache.spark.sql.catalyst.expressions.SortOrder],
    classOf[org.apache.spark.sql.catalyst.expressions.UnsafeRow],
    classOf[org.apache.spark.sql.catalyst.expressions.codegen.LazilyGeneratedOrdering],
    classOf[org.apache.spark.sql.catalyst.trees.Origin],
    classOf[org.apache.spark.sql.catalyst.util.QuantileSummaries],
    classOf[org.apache.spark.sql.execution.datasources.BasicWriteTaskStats],
    classOf[org.apache.spark.sql.execution.datasources.ExecutedWriteSummary],
    classOf[org.apache.spark.sql.types.ArrayType],
    classOf[org.apache.spark.sql.types.DataType],
    classOf[org.apache.spark.sql.types.Metadata],
    classOf[org.apache.spark.sql.types.StructField],
    classOf[org.apache.spark.sql.types.StructType],
    Class.forName("com.databricks.spark.avro.DefaultSource$SerializableConfiguration"),
    Class.forName("org.apache.spark.broadcast.TorrentBroadcast"),
    Class.forName("org.apache.spark.ml.classification.MultiClassSummarizer"),
    Class.forName("org.apache.spark.ml.feature.LabeledPoint"),
    Class.forName("org.apache.spark.ml.linalg.MatrixUDT"),
    Class.forName("org.apache.spark.ml.linalg.VectorUDT"),
    Class.forName("org.apache.spark.ml.optim.WeightedLeastSquares$Aggregator"),
    Class.forName("org.apache.spark.ml.optim.aggregator.HingeAggregator"),
    Class.forName("org.apache.spark.ml.optim.aggregator.LogisticAggregator"),
    Class.forName("org.apache.spark.ml.param.ParamValidators$$anonfun$alwaysTrue$1"),
    Class.forName("org.apache.spark.ml.tree.impl.DecisionTreeMetadata"),
    Class.forName("org.apache.spark.ml.tree.impl.DTStatsAggregator"),
    Class.forName("org.apache.spark.mllib.evaluation.binary.BinaryLabelCounter"),
    Class.forName("org.apache.spark.mllib.tree.configuration.QuantileStrategy$"),
    Class.forName("org.apache.spark.mllib.tree.impurity.EntropyAggregator"),
    Class.forName("org.apache.spark.mllib.tree.impurity.Gini$"),
    Class.forName("org.apache.spark.mllib.tree.impurity.GiniAggregator"),
    Class.forName("org.apache.spark.mllib.tree.impurity.GiniCalculator"),
    Class.forName("org.apache.spark.mllib.tree.impurity.Variance$"),
    Class.forName("org.apache.spark.mllib.tree.impurity.VarianceAggregator"),
    Class.forName("org.apache.spark.mllib.tree.impurity.VarianceCalculator"),
    Class.forName("org.apache.spark.mllib.tree.model.ImpurityStats"),
    Class.forName("org.apache.spark.sql.execution.datasources.FileFormatWriter$WriteTaskResult"),
    Class.forName("org.apache.spark.storage.BroadcastBlockId"),
    Class.forName("org.apache.spark.ml.param.shared.HasCheckpointInterval$$anonfun$1"),
    org.apache.spark.sql.catalyst.expressions.Ascending.getClass,
    org.apache.spark.sql.catalyst.expressions.Descending.getClass,
    org.apache.spark.sql.catalyst.expressions.NullsFirst.getClass,
    org.apache.spark.sql.catalyst.expressions.NullsLast.getClass,
    org.apache.spark.sql.types.BinaryType.getClass,
    org.apache.spark.sql.types.BooleanType.getClass,
    org.apache.spark.sql.types.ByteType.getClass,
    org.apache.spark.sql.types.CalendarIntervalType.getClass,
    org.apache.spark.sql.types.CharType.getClass,
    org.apache.spark.sql.types.DoubleType.getClass,
    org.apache.spark.sql.types.FloatType.getClass,
    org.apache.spark.sql.types.IntegerType.getClass,
    org.apache.spark.sql.types.LongType.getClass,
    org.apache.spark.sql.types.MapType.getClass,
    org.apache.spark.sql.types.NullType.getClass,
    org.apache.spark.sql.types.ObjectType.getClass,
    org.apache.spark.sql.types.ShortType.getClass,
    org.apache.spark.sql.types.StringType.getClass,
    org.apache.spark.sql.types.TimestampType.getClass,
    org.apache.spark.sql.types.VarcharType.getClass)
}
