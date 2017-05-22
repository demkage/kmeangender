package com.eter.spark.app.kmeangender;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Created by rusifer on 5/21/17.
 */
public class KMeanGender {
    private static final Logger log = LoggerFactory.getLogger(KMeanGender.class);
    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            log.error("Can't find argument for model output");
            log.debug("Actual arguments length: " + args.length);
            log.info("Use <application-name> path/to/model");
            return;
        }

        String output = args[0];

        SparkSession session = new SparkSession.Builder()
                .appName("KMeanGender")
                .config("spark.sql.hive.metastore.version", "3.0.0")
                .config("spark.sql.hive.metastore.jars", "/usr/local/hadoop/share/hadoop/yarn/*:" +
                        "/usr/local/hadoop/share/hadoop/yarn/lib/*:" +
                        "/usr/local/hadoop/share/mapreduce/lib/*:" +
                        "/usr/local/hadoop/share/hadoop/mapreduce/*:" +
                        "/usr/local/hadoop/share/hadoop/common/*:" +
                        "/usr/local/hadoop/share/hadoop/hdfs/*:" +
                        "/usr//local/hadoop/etc/hadoop:" +
                        "/usr/local/hadoop/share/hadoop/common/lib/*:" +
                        "/usr/local/hadoop/share/hadoop/common/*:" +
                        "/usr/local/hive/lib/*:")
                .enableHiveSupport()
                .getOrCreate();

        Pipeline pipeline = new Pipeline();

        Dataset<Row> dataset = session.sql("select customerid, sex from customersdetail");
        StringIndexer sexIndexer = new StringIndexer()
                .setInputCol("sex")
                .setOutputCol("sexIndex");
        OneHotEncoder sexEncoder = new OneHotEncoder()
                .setInputCol("sexIndex")
                .setOutputCol("features");

        KMeans kMeans = new KMeans()
                .setK(2)
                .setFeaturesCol("features")
                .setPredictionCol("prediction");

        pipeline.setStages(new PipelineStage[] {sexIndexer, sexEncoder, kMeans});

        PipelineModel pipelineModel = pipeline.fit(dataset);

        pipelineModel.save(output);
    }
}
