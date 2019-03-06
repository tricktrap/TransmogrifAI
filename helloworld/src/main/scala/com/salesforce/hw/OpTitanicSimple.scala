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
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types._
import com.salesforce.op.readers.DataReaders
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelSelector
import com.salesforce.op.stages.impl.classification.BinaryClassificationModelsToTry
import com.salesforce.op.stages.impl.classification.{OpLogisticRegression, OpRandomForestClassifier}
import com.salesforce.op.stages.impl.tuning.DataSplitter

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
 * Define a case class corresponding to our data file (nullable columns must be Option types)
 *
 * @param id       passenger id
 * @param survived 1: survived, 0: did not survive
 * @param pClass   passenger class
 * @param name     passenger name
 * @param sex      passenger sex (male/female)
 * @param age      passenger age (one person has a non-integer age so this must be a double)
 * @param sibSp    number of siblings/spouses traveling with this passenger
 * @param parCh    number of parents/children traveling with this passenger
 * @param ticket   ticket id string
 * @param fare     ticket price
 * @param cabin    cabin id string
 * @param embarked location where passenger embarked
 */
case class Passenger
(
  id: Int,
  survived: Int,
  pClass: Option[Int],
  name: Option[String],
  sex: Option[String],
  age: Option[Double],
  sibSp: Option[Int],
  parCh: Option[Int],
  ticket: Option[String],
  fare: Option[Double],
  cabin: Option[String],
  embarked: Option[String]
)

/**
 * A simplified TransmogrifAI example classification app using the Titanic dataset
 */
object OpTitanicSimple {

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
    val survived = FeatureBuilder.RealNN[Passenger].extract(_.survived.toRealNN).asResponse
    val pClass = FeatureBuilder.PickList[Passenger].extract(_.pClass.map(_.toString).toPickList).asPredictor
    val name = FeatureBuilder.Text[Passenger].extract(_.name.toText).asPredictor
    val sex = FeatureBuilder.PickList[Passenger].extract(_.sex.map(_.toString).toPickList).asPredictor
    val age = FeatureBuilder.Real[Passenger].extract(_.age.toReal).asPredictor
    val sibSp = FeatureBuilder.Integral[Passenger].extract(_.sibSp.toIntegral).asPredictor
    val parCh = FeatureBuilder.Integral[Passenger].extract(_.parCh.toIntegral).asPredictor
    val ticket = FeatureBuilder.PickList[Passenger].extract(_.ticket.map(_.toString).toPickList).asPredictor
    val fare = FeatureBuilder.Real[Passenger].extract(_.fare.toReal).asPredictor
    val cabin = FeatureBuilder.PickList[Passenger].extract(_.cabin.map(_.toString).toPickList).asPredictor
    val embarked = FeatureBuilder.PickList[Passenger].extract(_.embarked.map(_.toString).toPickList).asPredictor

    ////////////////////////////////////////////////////////////////////////////////
    // TRANSFORMED FEATURES
    /////////////////////////////////////////////////////////////////////////////////

    // Do some basic feature engineering using knowledge of the underlying dataset
    val familySize = sibSp + parCh + 1
    val estimatedCostOfTickets = familySize * fare
    val pivotedSex = sex.pivot()
    val normedAge = age.fillMissingWithMean().zNormalize()
    val ageGroup = age.map[PickList](_.value.map(v => if (v > 18) "adult" else "child").toPickList)

    // Define a feature of type vector containing all the predictors you'd like to use
    val passengerFeatures = Seq(
      pClass, name, age, sibSp, parCh, ticket,
      cabin, embarked, familySize, estimatedCostOfTickets,
      pivotedSex, ageGroup, normedAge
    ).transmogrify()

    // Optionally check the features with a sanity checker
    val checkedFeatures = survived.sanityCheck(passengerFeatures, removeBadFeatures = true)

    // Define the model we want to use (here a simple logistic regression) and get the resulting output
    val prediction = BinaryClassificationModelSelector.withTrainValidationSplit(
      modelTypesToUse = Seq(BinaryClassificationModelsToTry.OpLogisticRegression)
    ).setInput(survived, checkedFeatures).getOutput()

    val evaluator = Evaluators.BinaryClassification().setLabelCol(survived).setPredictionCol(prediction)

    ////////////////////////////////////////////////////////////////////////////////
    // WORKFLOW
    /////////////////////////////////////////////////////////////////////////////////

    // Define a way to read data into our Passenger class from our CSV file
    val dataReader = DataReaders.Simple.csvCase[Passenger](path = Option(csvFilePath), key = _.id.toString)

    // Define a new workflow and attach our data reader
    val workflow = new OpWorkflow().setResultFeatures(survived, prediction).setReader(dataReader)

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

    /*
    val permutationFeatureImportances = rawFeatureNamesToPermute.map{ rf =>
      val permutedDf = model.computeDataUpToAndPermute(prediction, nameToPermute = rf)
      val permutedMetric = evaluator.evaluate(permutedDf)
      println(s"originalMetric: ${fullMetric}")
      println(s"permutedMetric: ${permutedMetric}")

      rf -> (fullMetric - permutedMetric)
    }.toMap[String, Double]

    println(s"permutationFeatureImportances: ")
    permutationFeatureImportances.toSeq.sortBy(-_._2).foreach(println)
    */

    val allFeatures = Seq(
      pClass, name, age, sibSp, parCh, ticket,
      cabin, embarked, familySize, estimatedCostOfTickets,
      pivotedSex, ageGroup, normedAge
    )
    val rawFeatureNamesToDrop: Seq[String] = model.rawFeatures.filterNot(_.isResponse).map(_.name)

    var droppedFeatImp = rawFeatureNamesToDrop.map{feature_i =>
      val droppedFeatureVector = allFeatures.filterNot(_.name == feature_i).transmogrify()
      val droppedPrediction = modelInsights.selectedModelInfo.get.bestModelType match {
        case "OpLogisticRegression" =>
          new OpLogisticRegression().setInput(survived, droppedFeatureVector).getOutput()
        case "OpRandomForestClassifier" =>
          new OpRandomForestClassifier().setInput(survived, droppedFeatureVector).getOutput()
      }
      val droppedWorkflow = new OpWorkflow().setResultFeatures(survived, droppedPrediction).setReader(dataReader)
      val droppedModel = droppedWorkflow.train()
      println(s"Model summary:\n${droppedModel.summaryPretty()}")
      println("Scoring the model")
      val droppedEvaluator = Evaluators.BinaryClassification().setLabelCol(survived).setPredictionCol(droppedPrediction)
      val (droppedScores, droppedMetrics) = droppedModel.scoreAndEvaluate(evaluator = droppedEvaluator)
      println(s"Metrics: $droppedMetrics")
      val droppedDf = droppedModel.computeDataUpTo(droppedPrediction)
      val droppedMetrics2 = droppedEvaluator.evaluate(droppedDf)
      feature_i -> (fullMetric - droppedMetrics2)
    }.toMap[String, Double]
    println(s"droppedFeatureImportances: ")
    droppedFeatImp.toSeq.sortBy(-_._2).foreach(println)

    // test push
    // val (fullTrainDf, fullHoldoutDf) = dataSplitter.split(fullDf)
    // val trainSetMetric = evaluator.evaluate(fullTrainDf)
    // val holdoutSetMetric = evaluator.evaluate(fullHoldoutDf)

    // Stop Spark gracefully
    spark.stop()
  }
}
