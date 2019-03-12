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

package com.salesforce.hw

import com.salesforce.op._
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.{FeatureBuilder, FeatureLike}
import com.salesforce.op.features.types._
import com.salesforce.op.readers.{CSVProductReader, DataReaders}
import com.salesforce.op.stages.impl.regression.RegressionModelsToTry.OpRandomForestRegressor
import com.salesforce.op.stages.impl.regression._
import com.salesforce.op.stages.impl.tuning.DataSplitter
import com.salesforce.op.testkit._
import com.salesforce.op.utils.io.csv.CSVOptions
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

case class BostonHouse2
(
  rowId: Int,
  crim: Double,
  zn: Double,
  indus: Double,
  chas: String,
  nox: Double,
  rm: Double,
  age: Double,
  dis: Double,
  rad: Int,
  tax: Double,
  ptratio: Double,
  b: Double,
  lstat: Double,
  medv: Double
)

/**
 * A simplified TransmogrifAI example classification app using the Titanic dataset
 */
object OpBostonSimple {

  /**
   * Run this from the command line with
   * ./gradlew sparkSubmit -Dmain=com.salesforce.hw.OpTitanicSimple -Dargs=/full/path/to/csv/file
   */
  def main(args: Array[String]): Unit = {
    if (args.isEmpty) {
      println("You need to pass in the CSV file path as an argument")
      sys.exit(1)
    }
    val csvFilePath = args(0)
    println(s"Using user-supplied CSV file path: $csvFilePath")

    // Set up a SparkSession as normal
    implicit val spark = SparkSession.builder.config(new SparkConf()).getOrCreate()
    import spark.implicits._ // Needed for Encoders for the Passenger case class



    ////////////////////////////////////////////////////////////////////////////////
    // RAW FEATURE DEFINITIONS
    /////////////////////////////////////////////////////////////////////////////////

    // Define features using the OP types based on the data
    val rowId = FeatureBuilder.Integral[BostonHouse2].extract(_.rowId.toIntegral).asPredictor
    val crim = FeatureBuilder.RealNN[BostonHouse2].extract(_.crim.toRealNN).asPredictor
    val zn = FeatureBuilder.RealNN[BostonHouse2].extract(_.zn.toRealNN).asPredictor
    val indus = FeatureBuilder.RealNN[BostonHouse2].extract(_.indus.toRealNN).asPredictor
    val chas = FeatureBuilder.PickList[BostonHouse2].extract(x => Option(x.chas).toPickList).asPredictor
    val nox = FeatureBuilder.RealNN[BostonHouse2].extract(_.nox.toRealNN).asPredictor
    val rm = FeatureBuilder.RealNN[BostonHouse2].extract(_.rm.toRealNN).asPredictor
    val age = FeatureBuilder.RealNN[BostonHouse2].extract(_.age.toRealNN).asPredictor
    val dis = FeatureBuilder.RealNN[BostonHouse2].extract(_.dis.toRealNN).asPredictor
    val rad = FeatureBuilder.Integral[BostonHouse2].extract(_.rad.toIntegral).asPredictor
    val tax = FeatureBuilder.RealNN[BostonHouse2].extract(_.tax.toRealNN).asPredictor
    val ptratio = FeatureBuilder.RealNN[BostonHouse2].extract(_.ptratio.toRealNN).asPredictor
    val b = FeatureBuilder.RealNN[BostonHouse2].extract(_.b.toRealNN).asPredictor
    val lstat = FeatureBuilder.RealNN[BostonHouse2].extract(_.lstat.toRealNN).asPredictor

    val medv = FeatureBuilder.RealNN[BostonHouse2].extract(_.medv.toRealNN).asResponse

    val randomPickListData = FeatureBuilder.PickList[BostonHouse2]
      .extract(_ => {
        val pickListData = RandomText.pickLists(domain = List("A", "B", "C", "D", "E", "F", "G", "H", "I"))
          .withProbabilityOfEmpty(0.2)
        pickListData.next()
      }).asPredictor

    val randomNumericData = FeatureBuilder.Real[BostonHouse2]
      .extract(_ => {
        val numericData = RandomReal.logNormal[Real](mean = 10.0, sigma = 1.0)
          .withProbabilityOfEmpty(0.2)
        numericData.next()
      }).asPredictor

    ////////////////////////////////////////////////////////////////////////////////
    // TRANSFORMED FEATURES
    /////////////////////////////////////////////////////////////////////////////////

    // Define a feature of type vector containing all the predictors you'd like to use
    val houseFeatures = Seq(rowId, crim, zn, indus, chas, nox, rm, age,
      dis, rad, tax, ptratio, b, lstat, randomPickListData, randomNumericData
    ).transmogrify()

    // Optionally check the features with a sanity checker
    val checkedFeatures = medv.sanityCheck(houseFeatures, removeBadFeatures = true)

    // Define the model we want to use (here a simple logistic regression) and get the resulting output
    val prediction: FeatureLike[Prediction] = RegressionModelSelector.withTrainValidationSplit(
      modelTypesToUse = Seq(RegressionModelsToTry.OpRandomForestRegressor)
    ).setInput(medv, checkedFeatures).getOutput()

    // val prediction = new OpLinearRegression().setInput(medv, houseFeatures).getOutput()
    // val prediction = new OpRandomForestRegressor().setInput(medv, houseFeatures).getOutput()
    // val prediction = new OpGBTRegressor().setInput(medv, houseFeatures).getOutput()

    ////////////////////////////////////////////////////////////////////////////////
    // WORKFLOW
    /////////////////////////////////////////////////////////////////////////////////

    // Define a way to read data into our Passenger class from our CSV file
    val dataReader = new CSVProductReader[BostonHouse2](readPath = Option(csvFilePath),
      key = _.rowId.toString, options = new CSVOptions(header = true))

    // Define a new workflow and attach our data reader
    val workflow = new OpWorkflow().setResultFeatures(medv, prediction).setReader(dataReader)

    val evaluator = Evaluators.Regression().setLabelCol(medv).setPredictionCol(prediction)

    // Fit the workflow to the data
    val model = workflow.train()
    println(s"Model summary:\n${model.summaryPretty()}")

    // Manifest the result features of the workflow
    println("Scoring the model")
    val (scores, metrics) = model.scoreAndEvaluate(evaluator = evaluator)

    println("Metrics:\n" + metrics)

    val modelInsights = model.modelInsights(prediction)
    val dataSplitter = DataSplitter(
      seed = modelInsights.selectedModelInfo.get.dataPrepParameters("seed").asInstanceOf[Long],
      reserveTestFraction = modelInsights.selectedModelInfo.get
        .dataPrepParameters("reserveTestFraction").asInstanceOf[Double]
    )

    val fullDf = model.computeDataUpTo(prediction)
    val (trainDf, holdoutDf) = dataSplitter.split(fullDf)
    println(s"Calculate metrics on the full dataframe, the training set, and the holdout set")
    val trainMetric = evaluator.evaluate(trainDf)
    val holdoutMetric = evaluator.evaluate(holdoutDf)
    val fullMetric = evaluator.evaluate(fullDf)
    println(s"trainMetric: $trainMetric, holdoutMetric: $holdoutMetric, fullMetric: $fullMetric")

    println()
    println(s"Calculating permutation feature importances...")

    val rawFeatureNamesToPermute: Seq[String] = model.rawFeatures.filterNot(_.isResponse).map(_.name)
    println(s"rawFeatures: ${model.rawFeatures.toList}")

    // TODO: Split off holdout set and only permute columns there (much cheaper!).
    // Is that possible with computeDataUpTo?

    // TODO: Add some features in the Titanic dataset that shouldn't be good predictors (eg. random numerical and
    // categorical features)

    // TODO: Show that random forest, logistic regression, naive bayes, svm, gradient boosted trees, etc. all give
    // similar and comparable feature importances to each other with permutation feature importances, while the
    // model-specific measures are hard to compare / inconsistent

    // TODO: Compare the permutation feature importances and model-specific feature importances to drop-column
    // feature importances (ground truth)

    val permutationFeatureImportances = rawFeatureNamesToPermute.map{ rf =>
      val permutedDf = model.computeDataUpToAndPermute(prediction, nameToPermute = rf)
      val (permutedTrainDf, permutedHoldoutDf) = dataSplitter.split(permutedDf)
      val permutedTrainMetric = evaluator.evaluate(permutedTrainDf)
      val permutedHoldoutMetric = evaluator.evaluate(permutedHoldoutDf)

      println(s"raw feature: $rf")
      println(s"trainMetric: $trainMetric")
      println(s"holdoutMetric: $holdoutMetric")
      println(s"permutedTrainMetric: $permutedTrainMetric")
      println(s"permutedHoldoutMetric: $permutedHoldoutMetric")
      println()

      // Switch these because the metric we're using RMSE is better the smaller it is
      rf -> -(trainMetric - permutedTrainMetric)
    }.toMap[String, Double]

    println(s"permutationFeatureImportances: ")
    permutationFeatureImportances.toSeq.sortBy(-_._2).foreach(println)
    println()

    val pfiSum = permutationFeatureImportances.values.map(math.abs).sum
    val normalizedPFIs = permutationFeatureImportances.mapValues(_ / pfiSum)

    println(s"normalized permutation feature importances: ")
    normalizedPFIs.toSeq.sortBy(-_._2).foreach(println)

    println()
    println(s"top 40 model-dependent feature contributions:")
    modelInsights.features.flatMap(_.derivedFeatures)
      .sortBy(x => -math.abs(x.contribution.headOption.getOrElse(0.0))).take(50)
      .foreach(x => println(x.derivedFeatureName, x.corr, x.contribution.headOption.getOrElse(0.0)))

    // Correlation between feature importances and correlations
    val contributionMap: Map[String, (Double, Double)] = modelInsights.features.flatMap(_.derivedFeatures)
      .map(x => x.derivedFeatureName -> (x.corr.getOrElse(0.0), x.contribution.headOption.getOrElse(0.0))).toMap



    // Stop Spark gracefully
    spark.stop()
  }
}
